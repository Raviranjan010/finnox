import os
import logging
from typing import List
from celery import Celery
from finbert.finbert import SentimentEngine
from finbert.ner import EntitySentimentTracker
from finbert.explain import SentimentExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Celery with redis broker and backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
broker = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

broker.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Load pipeline components lazily when task starts
MODEL_PATH = os.getenv("MODEL_PATH", "/src/models/classifier_model/finbert-sentiment")
model = None
tracker = None
explainer = None

def get_model():
    global model
    if model is None:
        logger.info("Loading SentimentEngine from %s...", MODEL_PATH)
        model = SentimentEngine(MODEL_PATH)
    return model

def get_tracker():
    global tracker
    if tracker is None:
        logger.info("Initializing EntitySentimentTracker...")
        tracker = EntitySentimentTracker(get_model())
    return tracker

def get_explainer():
    global explainer
    if explainer is None:
        logger.info("Initializing SentimentExplainer...")
        engine = get_model()
        explainer = SentimentExplainer(engine.model, engine.tokenizer)
    return explainer

@broker.task(name="infer.batch")
def infer_batch(texts: List[str]):
    """Async batch task to run sentiment predictions, NER, and SHAP explainability on a list of texts."""
    sentiment_model = get_model()
    entity_tracker = get_tracker()
    sentiment_explainer = get_explainer()
    
    results = []
    for text in texts:
        # 1. Predict sentence-level sentiments
        pred_df = sentiment_model.predict(text)
        predictions = pred_df.to_dict(orient='records')
        
        # Determine overall document-level sentiment from predictions
        best_pred = {}
        if predictions:
            # We treat the highest confidence sentiment prediction as the document baseline
            best_pred = max(predictions, key=lambda x: x.get("confidence", 0.0))
            
        doc_sentiment = {
            "label": best_pred.get("prediction", "neutral"),
            "confidence": best_pred.get("confidence", 1.0)
        }
        
        # 2. Extract entity sentiment mapping
        entities = entity_tracker.extract_entity_sentiment(text, doc_sentiment)
        
        # 3. Generate SHAP token importance scores and explanation html
        explanation = sentiment_explainer.explain(text)
        
        # 4. Collate results
        results.append({
            "predictions": predictions,
            "entities": entities,
            "explanation": explanation
        })
        
    return results
