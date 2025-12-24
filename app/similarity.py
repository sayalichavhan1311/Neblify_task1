import csv
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use absolute paths
TRANSACTIONS_FILE = r"d:\Challenge_Task\Transaction_matcher1\data\transactions.csv"


def clean_text(text):
    """Clean text: lowercase, remove refs, keep only letters and spaces"""
    text = text.lower()
    text = re.sub(r"ref.*", "", text)       # remove ref part
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters and spaces
    return " ".join(text.split())            # remove extra spaces


def load_transactions():
    """Load all transactions and return IDs and cleaned descriptions"""
    ids = []
    texts = []

    with open(TRANSACTIONS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            texts.append(clean_text(row["description"]))

    return ids, texts


def find_similar_transactions(transaction_id, top_k=3):
    """Find top_k similar transactions using TF-IDF + cosine similarity"""
    ids, texts = load_transactions()

    if transaction_id not in ids:
        return None

    # Convert descriptions to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    # Get index of current transaction
    idx = ids.index(transaction_id)

    # Calculate cosine similarity with all transactions
    similarities = cosine_similarity(vectors[idx], vectors)[0]

    # Get top_k most similar (excluding the transaction itself)
    similar_indices = similarities.argsort()[::-1][1:top_k+1]

    return [ids[i] for i in similar_indices]
