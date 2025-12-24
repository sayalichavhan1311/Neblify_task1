from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from app.matcher import match_users
from app.similarity import find_similar_transactions
import os

app = FastAPI(
    title="Deel Python AI Engineer Challenge API",
    version="1.0.0",
    description="Transaction matching, similarity search, and scalability analysis"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchUser(BaseModel):
    """User match response"""
    id: str
    match_metric: float


class MatchResponse(BaseModel):
    """Response for matching users"""
    users: List[MatchUser]
    total_number_of_matches: int


class SimilarityRequest(BaseModel):
    """Request body for finding similar transactions"""
    transaction_id: str
    top_k: int = 3


class SimilarityResponse(BaseModel):
    """Response for similar transactions"""
    similar_transactions: List[str]


@app.get("/", response_class=HTMLResponse)
def root():
    # Return HTML UI if index.html exists
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    return """
    <html>
        <body>
            <h1>Transaction Matching API</h1>
            <p><a href="/docs">View API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/favicon.ico", response_class=Response)
def favicon():
    """Serve favicon (minimal 1x1 transparent GIF to prevent 404 errors)"""
    gif_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x01\x44\x00\x3b'
    return Response(content=gif_data, media_type="image/gif")



# TASK-1: MATCH USERS TO TRANSACTION

@app.get("/match_users/{transaction_id}", response_model=MatchResponse)
def match_users_api(transaction_id: str):
    """
    Task-1: Match a transaction to users based on description.
    Uses fuzzy string matching to handle typos and variations.
    
    Example: /match_users/caqjJtrI
    """
    result = match_users(transaction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result


# ============================================
# TASK-2: FIND SIMILAR TRANSACTIONS
# ============================================
@app.post("/similar_transactions", response_model=SimilarityResponse)
def similar_transactions_api(payload: SimilarityRequest):
    """
    Task-2: Find similar transactions using TF-IDF and cosine similarity.
    
    Example:
    {
        "transaction_id": "caqjJtrI",
        "top_k": 3
    }
    """
    result = find_similar_transactions(payload.transaction_id, payload.top_k)
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return {"similar_transactions": result if result else []}
