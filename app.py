# # import streamlit as st
# # import joblib
# # import os

# # BASE_DIR = os.path.dirname(__file__)

# # model_path = os.path.join(BASE_DIR, "spam_model.joblib")
# # vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

# # model = joblib.load(model_path)
# # vectorizer = joblib.load(vectorizer_path)

# # st.title("📧 Spam Mail Detector")

# # text = st.text_area("Enter email text")

# # if st.button("Predict"):
# #     if text.strip() == "":
# #         st.warning("Please enter some text")
# #     else:
# #         vec = vectorizer.transform([text])
# #         pred = model.predict(vec)[0]

# #         if pred == 1:
# #             st.error("🚨 Spam Mail")
# #         else:
# #             st.success("✅ Not Spam")



# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import os
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = os.path.dirname(__file__)

# model = joblib.load(os.path.join(BASE_DIR, "spam_model.joblib"))
# vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.joblib"))

# class Email(BaseModel):
#     text: str

# @app.get("/")
# def home():
#     return {"status": "API running"}

# @app.post("/predict")
# def predict(email: Email):
#     vec = vectorizer.transform([email.text])
#     pred = model.predict(vec)[0]
#     return {"prediction": int(pred)}
