from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator # type: ignore
import os
from db_models import SessionLocal, Registro

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

# --- Modelo HuggingFace ---
modelo_huggingface = "GABRYEL25770/TrainedModel"
tokenizer = T5Tokenizer.from_pretrained(modelo_huggingface)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tradutor ---
translator = Translator()

# --- Funções ---
def traduzir_para_ingles(texto_pt):
    traducao = translator.translate(texto_pt, src='pt', dest='en')
    return traducao.text

def predict_sentiment(text):
    texto_en = traduzir_para_ingles(text)

    input_text = f"classify sentiment: {texto_en}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)

    model = T5ForConditionalGeneration.from_pretrained(modelo_huggingface).to(device)

    if torch.cuda.is_available():
        model.half()

    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=20, num_beams=3)

    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()

    return sentiment

# --- Rota principal ---
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "Texto vazio"}), 400

    sentimento = predict_sentiment(user_text)

    return jsonify({"sentiment": sentimento})

@app.route("/save", methods=["POST", "OPTIONS"])
def save():
    if request.method == "OPTIONS":
        return '', 200  # responde ao preflight
    
    data = request.json
    texto = data.get("text", "")
    sentimento = data.get("sentiment", "")

    if not texto or not sentimento:
        return jsonify({"error": "Texto e sentimento são obrigatórios."}), 400

    db = SessionLocal()
    novo_registro = Registro(texto=texto, sentimento=sentimento)
    db.add(novo_registro)
    db.commit()
    db.close()

    return jsonify({"message": "Registro salvo com sucesso!"})

# --- Início ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
