import os
import time
from typing import List, Optional
import nltk
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from finbert.finbert import SentimentEngine
from finbert.ingestion import NewsIngestionPipeline
from finbert.ner import EntitySentimentTracker
from finbert.explain import SentimentExplainer
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

# Lazy loaded entities tracker & explainers
tracker = None
explainer = None

def get_tracker():
    global tracker
    if tracker is None:
        tracker = EntitySentimentTracker(model)
    return tracker

def get_explainer():
    global explainer
    if explainer is None:
        explainer = SentimentExplainer(model.model, model.tokenizer)
    return explainer

@app.post("/")
async def score(data: dict = Body(...)):
    """Predict sentiment of a single text with optional NER and SHAP explanations."""
    text = data.get('text')
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' key in request body")
    
    explain_req = data.get('explain', False)
    ner_req = data.get('ner', False)
    
    try:
        pred_df = model.predict(text)
        predictions = pred_df.to_dict(orient='records')
        
        if explain_req or ner_req:
            # Build doc-level sentiment based on highest-confidence sentence prediction
            best_pred = {}
            if predictions:
                best_pred = max(predictions, key=lambda x: x.get("confidence", 0.0))
            doc_sentiment = {
                "label": best_pred.get("prediction", "neutral"),
                "confidence": best_pred.get("confidence", 1.0)
            }
            
            response = {
                "predictions": predictions
            }
            if ner_req:
                response["entities"] = get_tracker().extract_entity_sentiment(text, doc_sentiment)
            if explain_req:
                response["explanation"] = get_explainer().explain(text)
            return response
            
        # Default list of predictions directly for strict backwards compatibility
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/news")
async def analyze_news(data: dict = Body(default={})):
    """Ingest articles for tickers and predict sentiments (with optional NER and SHAP)."""
    tickers = data.get('tickers', [])
    explain_req = data.get('explain', False)
    ner_req = data.get('ner', False)
    
    pipeline = NewsIngestionPipeline()
    try:
        articles = await pipeline.ingest(tickers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest news: {str(e)}")
        
    results = []
    for article in articles:
        pred_df = model.predict(article.text)
        predictions = pred_df.to_dict(orient='records')
        
        item = {
            "ticker_mentions": article.ticker_mentions,
            "source": article.source,
            "published": article.published,
            "text": article.text,
            "predictions": predictions
        }
        
        if explain_req or ner_req:
            best_pred = {}
            if predictions:
                best_pred = max(predictions, key=lambda x: x.get("confidence", 0.0))
            doc_sentiment = {
                "label": best_pred.get("prediction", "neutral"),
                "confidence": best_pred.get("confidence", 1.0)
            }
            
            if ner_req:
                item["entities"] = get_tracker().extract_entity_sentiment(article.text, doc_sentiment)
            if explain_req:
                item["explanation"] = get_explainer().explain(article.text)
                
        results.append(item)
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
