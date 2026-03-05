import firebase_admin
from firebase_admin import credentials, firestore

# Load Firebase service account key
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

# Connect to Firestore
db = firestore.client()

def get_db():
    return db
