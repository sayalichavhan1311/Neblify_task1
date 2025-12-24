#!/usr/bin/env python
import sys
import os

print(f"Working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

print("\nTesting imports...")
try:
    from app.matcher import find_transaction, match_users
    print("[OK] matcher imported successfully")
    
    print("\nTesting find_transaction('uU2NRqif')...")
    result = find_transaction("uU2NRqif")
    print(f"Result: {result}")
    
    if result:
        print("\n[OK] Transaction found! Testing match_users...")
        matches = match_users("uU2NRqif")
        print(f"Matches: {matches}")
    else:
        print("\n[FAIL] Transaction not found")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
