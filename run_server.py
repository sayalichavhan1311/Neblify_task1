#!/usr/bin/env python
"""
Transaction Matching API Server Launcher
Starts the FastAPI server and automatically opens it in the browser
"""

import sys
import os
import signal
import webbrowser
import time
import threading

def signal_handler(sig, frame):
    print(f'\n\n🛑 Server stopped by user (Ctrl+C)')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Configuration
HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"

print("\n" + "="*80)
print("  🚀 TRANSACTION MATCHING API - SERVER LAUNCHER".center(80))
print("="*80)
print()

print(f"📍 API URL (Web UI):    {URL}")
print(f"📊 API Documentation:   {URL}/docs")
print(f"🔌 API Endpoint (Task1): {URL}/match_users/{{transaction_id}}")
print(f"🔌 API Endpoint (Task2): {URL}/similar_transactions")
print()

print(f"📂 Working directory: {os.getcwd()}")
print(f"🐍 Python version: {sys.version.split()[0]}")
print()

# Import the app
print("⏳ Loading application...")
try:
    from app.main import app
    print("✅ Application loaded successfully\n")
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Function to open browser
def open_browser_async():
    """Open browser in background thread"""
    time.sleep(2)  # Wait for server to start
    print(f"🌍 Opening browser at {URL}...\n")
    try:
        webbrowser.open(URL)
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"   Please open manually: {URL}\n")

# Start browser in background thread
browser_thread = threading.Thread(target=open_browser_async, daemon=True)
browser_thread.start()

# Start the server
print("🔄 Starting server...\n")
print("-" * 80)

try:
    import uvicorn
    config = uvicorn.Config(
        app=app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
except KeyboardInterrupt:
    print(f"\n\n🛑 Server stopped by user (Ctrl+C)")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Server error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

