# backend/seed_db.py
"""
Run this script to seed Firestore with sample products, makeup_shades, and palettes.
python seed_db.py
"""
import random
import time
import uuid
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore
import os

SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT_JSON", "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    initialize_app(cred)

db = firestore.client()

NUM_PRODUCTS = 500  # increase to create a "big" dataset (careful with quota)
BRANDS = ["MockBrand", "JeansCo", "OuterLab", "UrbanFit", "StudioA", "PaletteCo"]
OCCASIONS = ["casual", "formal", "party", "office"]
BODY_TYPES = ["athletic", "slim", "curvy", "muscular"]

def create_products(n=NUM_PRODUCTS):
    col = db.collection("products")
    for i in range(n):
        title = f"Product {i+1} - {random.choice(['Chic','Bold','Classic','Modern'])}"
        price = round(random.uniform(15, 199), 2)
        brand = random.choice(BRANDS)
        tags = random.sample(OCCASIONS, k=random.randint(1,2))
        recommended_for = random.sample(BODY_TYPES, k=random.randint(1,2))
        doc = {
            "title": title,
            "price": price,
            "brand": brand,
            "imageUrl": f"https://picsum.photos/seed/{uuid.uuid4().hex}/600/400",
            "tags": tags,
            "recommended_for": recommended_for,
            "createdAt": firestore.SERVER_TIMESTAMP
        }
        col.add(doc)
        if (i+1) % 50 == 0:
            print(f"Created {i+1} products")
    print("Products seeding done.")

def create_makeup():
    col = db.collection("makeup_shades")
    examples = [
        {"name":"Nude Pink","type":"lipstick","skin_tones":["fair","medium"]},
        {"name":"Coral Red","type":"lipstick","skin_tones":["medium","olive"]},
        {"name":"Plum","type":"lipstick","skin_tones":["deep","olive"]},
        {"name":"Peach Glow","type":"blush","skin_tones":["fair","medium"]},
        {"name":"Warm Bronze","type":"bronzer","skin_tones":["medium","olive"]},
    ]
    for e in examples:
        col.add({**e, "createdAt": firestore.SERVER_TIMESTAMP})
    print("Makeup seeded.")

def create_palettes():
    col = db.collection("palettes")
    palettes = [
        {"name":"Warm Autumn", "skin_tones":["olive","medium"], "swatches":["#9B6B3F","#708238","#D9C08A","#6B8E23","#3A5A40"]},
        {"name":"Cool Winter", "skin_tones":["fair","medium"], "swatches":["#EFD9D1","#C7D6E8","#9BB7D4","#2B3A67","#1C1F2C"]},
        {"name":"Deep Jewel", "skin_tones":["deep"], "swatches":["#3B1F2B","#7A3B47","#B86B6B","#4A2D2A","#2F1A1A"]},
    ]
    for p in palettes:
        col.add({**p, "createdAt": firestore.SERVER_TIMESTAMP})
    print("Palettes seeded.")

if __name__ == "__main__":
    create_products()
    create_makeup()
    create_palettes()
    print("Seeding complete.")
