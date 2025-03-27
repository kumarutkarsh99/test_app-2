from flask import Flask, request, jsonify
import os
import io
import csv
import json
import nltk
import time
from transformers import pipeline
import torch
from flask_cors import CORS

# NLTK setup
NLTK_DATA_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
try:
    nltk.download("punkt", quiet=True, download_dir=NLTK_DATA_DIR)
except Exception as e:
    print(f"NLTK download error: {e}")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health():
    return "OK", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    sentences = []

    # --- Handle File Upload ---
    if request.files:
        file = request.files.get('file') or next(iter(request.files.values()), None)
        if not file:
            return jsonify({"error": "No file provided"}), 400
        filename = file.filename or ''
        content = file.read()

        if filename.lower().endswith('.csv') or 'csv' in (file.content_type or ''):
            content_str = content.decode('utf-8', errors='ignore')
            reader = csv.reader(io.StringIO(content_str))
            rows = list(reader)
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

            if isinstance(data, dict):
                if "sentences" in data and isinstance(data["sentences"], list):
                    for item in data["sentences"]:
                        if isinstance(item, str) and item.strip():
                            sentences.append(item.strip())
                        elif isinstance(item, dict):
                            val = item.get("text") or item.get("sentence")
                            if val and isinstance(val, str) and val.strip():
                                sentences.append(val.strip())
                elif "text" in data and isinstance(data["text"], list):
                    for item in data["text"]:
                        if isinstance(item, str) and item.strip():
                            sentences.append(item.strip())
                else:
                    for val in data.values():
                        if isinstance(val, str) and val.strip():
                            sentences.append(val.strip())
                        elif isinstance(val, list):
                            for item in val:
                                if isinstance(item, str) and item.strip():
                                    sentences.append(item.strip())
                                elif isinstance(item, dict):
                                    val = item.get("text") or item.get("sentence")
                                    if val and isinstance(val, str) and val.strip():
                                        sentences.append(val.strip())
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        sentences.append(item.strip())
                    elif isinstance(item, dict):
                        val = item.get("text") or item.get("sentence")
                        if val and isinstance(val, str) and val.strip():
                            sentences.append(val.strip())
        else:
            return jsonify({"error": "Unsupported file type"}), 400

    # --- Handle Raw Text Input ---
    else:
        text_input = request.form.get("text")
        if not text_input:
            return jsonify({"error": "No text provided"}), 400
        sentences = nltk.tokenize.sent_tokenize(text_input.strip())

    if not sentences:
        return jsonify({"error": "No sentences found in input"}), 400

    # --- Limit Sentence Count ---
    MAX_SENTENCES = 100
    if len(sentences) > MAX_SENTENCES:
        return jsonify({"error": f"Too many sentences ({len(sentences)}). Limit is {MAX_SENTENCES}."}), 400

    # --- Load model inside route to save memory ---
    try:
        model_name = "kumarutkarsh99/biasfree"
        classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {e}"}), 500

    # --- Per-sentence Inference (Slow, Safe) ---
    results = []
    for sent in sentences:
        try:
            output = classifier(sent)[0]
            label = output.get("label", "").lower()
            score = output.get("score", 0.0)

            if "unbias" in label or "no bias" in label or "not bias" in label:
                biased = False
            elif "bias" in label:
                biased = True
            else:
                biased = score >= 0.5

            results.append({
                "sentence": sent,
                "biased": bool(biased),
                "confidence": float(score)
            })

            time.sleep(0.25)  # Prevent memory spikes

        except Exception as e:
            return jsonify({"error": f"Inference failed on: '{sent}' — {e}"}), 500

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
