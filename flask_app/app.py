from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import pickle

app = Flask(__name__)
CORS(app)


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()

        comment = comment.strip()

        comment = re.sub(r'\n', ' ', comment)

        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        return comment


def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'comments' not in data:
        return jsonify({"error": "No comments Provided"}), 400
    comments = data.get('comments')

    if not comments:
        return jsonify({"error":"No comments Provided"}),400
    
    try:
        preprocess_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comment = vectorizer.transform(preprocess_comments)

        predictions = model.predict(transformed_comment)
        predictions = [int(p) for p in predictions]
    except Exception as e:
        return jsonify({"error":f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000)

