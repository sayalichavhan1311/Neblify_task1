import csv
import re
import os
from rapidfuzz import fuzz

# Use absolute paths
USERS_FILE = r"d:\Challenge_Task\Transaction_matcher1\data\users.csv"
TRANSACTIONS_FILE = r"d:\Challenge_Task\Transaction_matcher1\data\transactions.csv"
MATCH_THRESHOLD = 50  # Lowered from 60 to allow more fuzzy matches


# -------------------------------
# Read users.csv
# -------------------------------
def load_users():
    users = []
    with open(USERS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append({
                "id": row["id"],
                "name": row["name"].lower().strip()
            })
    return users


# -------------------------------
# Find transaction by ID
# -------------------------------
def find_transaction(transaction_id):
    print(f"\n[DEBUG] Finding transaction: {transaction_id}")
    print(f"[DEBUG] File path: {TRANSACTIONS_FILE}")
    print(f"[DEBUG] File exists: {os.path.exists(TRANSACTIONS_FILE)}")
    
    try:
        with open(TRANSACTIONS_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                current_id = row.get("id", "").strip()
                if current_id == transaction_id:
                    print(f"[DEBUG] Found transaction at row {row_count}")
                    return row
            
            print(f"[DEBUG] Transaction not found after reading {row_count} rows")
    except Exception as e:
        print(f"[ERROR] Error reading transactions file: {e}")
        import traceback
        traceback.print_exc()
    
    return None


# -------------------------------
# Clean description text
# (remove symbols, refs, numbers)
# -------------------------------
def clean_description(text):
    text = text.lower()
    text = re.sub(r"ref.*", "", text)       # remove ref part
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters
    text = re.sub(r"\s+", " ", text)        # remove extra spaces
    return text.strip()


# -------------------------------
# Match users with transaction
# -------------------------------
def match_users(transaction_id):
    print(f"\n[DEBUG] match_users called with: {transaction_id}")
    print(f"[DEBUG] TRANSACTIONS_FILE: {TRANSACTIONS_FILE}")
    print(f"[DEBUG] File exists: {os.path.exists(TRANSACTIONS_FILE)}")
    
    transaction = find_transaction(transaction_id)

    if not transaction:
        print(f"[DEBUG] Transaction not found!")
        return None
    
    print(f"[DEBUG] Transaction found: {transaction['id']}")

    description = clean_description(transaction["description"])
    print(f"[DEBUG] Cleaned description: '{description}'")
    users = load_users()
    print(f"[DEBUG] Loaded {len(users)} users")

    matches = []
    top_scores = []

    for user in users:
        score = fuzz.token_sort_ratio(description, user["name"])
        top_scores.append((score, user["name"]))

        if score >= MATCH_THRESHOLD:
            matches.append({
                "id": user["id"],
                "match_metric": score
            })

    top_scores.sort(reverse=True)
    print(f"[DEBUG] Top 5 scores: {top_scores[:5]}")
    print(f"[DEBUG] Threshold: {MATCH_THRESHOLD}")
    print(f"[DEBUG] Matches found: {len(matches)}")

    matches.sort(key=lambda x: x["match_metric"], reverse=True)

    return {
        "users": matches,
        "total_number_of_matches": len(matches)
    }
