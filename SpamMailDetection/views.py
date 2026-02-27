import numpy as np
import pandas as pd
import joblib
import re
import os

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Loading the model and vectorizer once

model_path = os.path.join(BASE_DIR,"spam_model.joblib")
model = joblib.load(model_path)

vectorizer_path = os.path.join(BASE_DIR,"tfidf_vectorizer.joblib")
vectorizer = joblib.load(vectorizer_path)


stopwords1 = set(stopwords.words("english"))

from django.shortcuts import render

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    words = text.split()
    words = [word for word in words if word not in stopwords1]
    
    return " ".join(words)





@api_view(['POST'])
def predict(request):
    mail = request.data.get('message', '')
    if not mail.strip():
        return Response({"error": "Empty message"}, status=400)

    cleaned_mail = clean_text(mail)
    vectorized_mail = vectorizer.transform([cleaned_mail])

    prediction_label = model.predict(vectorized_mail)[0]
    proba_all = model.predict_proba(vectorized_mail)[0]

    # Safe handling if model has only 1 class
    if isinstance(proba_all, (float, np.float32, np.float64)):
        proba = 1.0 if prediction_label == "spam" else 0.0
    else:
        spam_index = list(model.classes_).index("spam")
        proba = proba_all[spam_index]

    prediction = 1 if prediction_label == "spam" else 0

    return Response({
        "prediction": prediction,
        "confidence": float(proba)
    })
