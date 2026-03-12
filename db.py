import streamlit as st
from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate(dict(st.secrets["firebase"]))
initialize_app(cred)

db = firestore.client()

def get_db():
    return db
