from flask import Flask, request, jsonify
import os
import io
import csv
import json
import nltk
from transformers import pipeline
import torch

# Download NLTK 'punkt' tokenizer and related data (including 'punkt_tab' for newer versions)
NLTK_DATA_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
try:
    nltk.download("punkt", quiet=True, download_dir=NLTK_DATA_DIR)
    try:
        nltk.download("punkt_tab", quiet=True, download_dir=NLTK_DATA_DIR)
    except Exception:
        pass
except Exception as e:
    print(f"NLTK download error: {e}")

# Load the Hugging Face model pipeline for text classification
model_name = "kumarutkarsh99/biasfree"
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=device)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health():
    return "OK", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    sentences = []
    # Check if a file is provided in the request
    if request.files:
        file = request.files.get('file') or next(iter(request.files.values()), None)
        if not file:
            return jsonify({"error": "No file provided"}), 400
        filename = file.filename or ''
        content = file.read()
        # Determine file type (CSV or JSON) by extension or content type
        if filename.lower().endswith('.csv') or 'csv' in (file.content_type or ''):
            content_str = content.decode('utf-8', errors='ignore')
            reader = csv.reader(io.StringIO(content_str))
            rows = list(reader)
            # Skip header row if present
            if rows:
                first_cell = rows[0][0] if len(rows[0]) > 0 else ''
                if first_cell.strip().lower() in ('text', 'sentence', 'sentences', 'input'):
                    rows = rows[1:]
            for row in rows:
                if row and len(row) > 0:
                    sentence = str(row[0]).strip()
                    if sentence:
                        sentences.append(sentence)
        elif filename.lower().endswith('.json') or 'json' in (file.content_type or ''):
            content_str = content.decode('utf-8', errors='ignore')
            try:
                data = json.loads(content_str)
            except Exception:
                return jsonify({"error": "Invalid JSON file"}), 400
            # Extract sentences from JSON structure
            if isinstance(data, dict):
                if "sentences" in data and isinstance(data["sentences"], list):
                    for item in data["sentences"]:
                        if isinstance(item, str):
                            if item.strip():
                                sentences.append(item.strip())
                        elif isinstance(item, dict):
                            if "text" in item:
                                val = str(item["text"]).strip()
                                if val:
                                    sentences.append(val)
                            elif "sentence" in item:
                                val = str(item["sentence"]).strip()
                                if val:
                                    sentences.append(val)
                elif "text" in data and isinstance(data["text"], list):
                    for item in data["text"]:
                        if isinstance(item, str):
                            if item.strip():
                                sentences.append(item.strip())
                else:
                    for val in data.values():
                        if isinstance(val, str):
                            if val.strip():
                                sentences.append(val.strip())
                        elif isinstance(val, list):
                            for item in val:
                                if isinstance(item, str):
                                    if item.strip():
                                        sentences.append(item.strip())
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        if item.strip():
                            sentences.append(item.strip())
                    elif isinstance(item, dict):
                        if "text" in item:
                            val = str(item["text"]).strip()
                            if val:
                                sentences.append(val)
                        elif "sentence" in item:
                            val = str(item["sentence"]).strip()
                            if val:
                                sentences.append(val)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    else:
        # Handle raw text input (JSON or form data)
        text_input = None
        if request.is_json:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            text_input = data.get("text")
        if text_input is None:
            text_input = request.form.get("text")
        if text_input is None or text_input == "":
            return jsonify({"error": "No text provided"}), 400
        text_input = str(text_input)
        sentences = nltk.tokenize.sent_tokenize(text_input)
    if not sentences:
        return jsonify({"error": "No sentences found in input"}), 400

    results = []
    for sent in sentences:
        try:
            output = classifier(sent)
        except Exception as e:
            return jsonify({"error": f"Model inference failed: {e}"}), 500
        # Pipeline returns a list for each input; take the first result dict
        result = output[0] if isinstance(output, list) else output
        label = str(result.get("label", ""))
        score = result.get("score", None)
        low_label = label.lower()
        # Determine bias based on label or score
        if "unbias" in low_label or "no bias" in low_label or "not bias" in low_label:
            biased_flag = False
        elif label.upper().startswith("LABEL_"):
            # If label is like "LABEL_0", "LABEL_1", etc.
            try:
                idx = int(label.split("_")[1])
                biased_flag = (idx != 0)
            except:
                biased_flag = True if "1" in label else False
        elif "bias" in low_label:
            # Label explicitly indicates bias (e.g., contains "bias")
            biased_flag = True
        elif score is not None and isinstance(score, (float, int)):
            # Fallback: use score threshold 0.5 if label name gives no clue
            biased_flag = True if score >= 0.5 else False
        else:
            biased_flag = False
        results.append({
            "sentence": sent,
            "biased": biased_flag,
            "confidence": float(score) if score is not None else None
        })
    return jsonify({"results": results}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  
    app.run(host='0.0.0.0', port=port)

