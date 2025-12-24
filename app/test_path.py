import os
import sys
import csv

# Print current info
print(f"Script location: {__file__}")
print(f"Script directory: {os.path.dirname(__file__)}")

# Get the absolute path to the data directory (from app folder, go up one level)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"BASE_DIR: {BASE_DIR}")

TRANSACTIONS_FILE = os.path.join(BASE_DIR, "data", "transactions.csv")
print(f"TRANSACTIONS_FILE: {TRANSACTIONS_FILE}")
print(f"File exists: {os.path.exists(TRANSACTIONS_FILE)}")

# Try to read
if os.path.exists(TRANSACTIONS_FILE):
    with open(TRANSACTIONS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            if count == 1:
                print(f"\nFirst transaction:")
                print(f"  ID: {row['id']}")
                print(f"  Description: {row['description']}")
        print(f"Total: {count} transactions")
