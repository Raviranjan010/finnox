import os
import time
from typing import List, Optional
import nltk
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from finbert.finbert import SentimentEngine
from finbert.ingestion import NewsIngestionPipeline
from tasks import broker

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

app = FastAPI(title="FinSential API", version="2.0")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model for synchronous / fallback endpoints
# Fallback to roberta-base is handled internally in SentimentEngine
MODEL_PATH = os.getenv("MODEL_PATH", "/src/models/classifier_model/finbert-sentiment")
model = SentimentEngine(MODEL_PATH)

@app.post("/")
async def score(data: dict = Body(...)):
    """Predict sentiment of a single text (synchronous, backwards compatible)."""
    text = data.get('text')
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' key in request body")
    try:
        pred_df = model.predict(text)
        return pred_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/news")
async def analyze_news(data: dict = Body(default={})):
    """Ingest articles for tickers and predict sentiments (async I/O, sync inference)."""
    tickers = data.get('tickers', [])
    pipeline = NewsIngestionPipeline()
    try:
        articles = await pipeline.ingest(tickers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest news: {str(e)}")
        
    results = []
    for article in articles:
        pred_df = model.predict(article.text)
        predictions = pred_df.to_dict(orient='records')
        results.append({
            "ticker_mentions": article.ticker_mentions,
            "source": article.source,
            "published": article.published,
            "text": article.text,
            "predictions": predictions
        })
    return {"results": results}

@app.post("/analyze/batch")
async def analyze_batch(texts: List[str] = Body(...), background_tasks: BackgroundTasks = None):
    """Async batch — handles 100s of articles; finBERT does 1 at a time"""
    task = broker.send_task("infer.batch", args=[texts])
    return {"task_id": task.id, "status": "queued"}

@app.get("/analyze/{task_id}")
async def get_result(task_id: str):
    result = broker.AsyncResult(task_id)
    return {"status": result.state, "result": result.result}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
