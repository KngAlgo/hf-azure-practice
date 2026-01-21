from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

app = Flask(__name__)

# Load model from YOUR Hugging Face repo
tokenizer = AutoTokenizer.from_pretrained("KNGTech/demov2")
model = TFAutoModelForSequenceClassification.from_pretrained("KNGTech/demov2")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = int(tf.argmax(outputs.logits, axis=-1)[0])
    
    return jsonify({
        "label": prediction,
        "sentiment": "positive" if prediction == 1 else "negative"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)