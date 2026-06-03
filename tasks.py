import os
import logging
from typing import List
from celery import Celery
from finbert.finbert import SentimentEngine

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

# Load the model lazily when task starts
MODEL_PATH = os.getenv("MODEL_PATH", "/src/models/classifier_model/finbert-sentiment")
model = None

def get_model():
    global model
    if model is None:
        logger.info("Loading SentimentEngine from %s...", MODEL_PATH)
        model = SentimentEngine(MODEL_PATH)
    return model

@broker.task(name="infer.batch")
def infer_batch(texts: List[str]):
    """Async batch task to run sentiment predictions on a list of texts."""
    sentiment_model = get_model()
    results = []
    for text in texts:
        pred_df = sentiment_model.predict(text)
        predictions = pred_df.to_dict(orient='records')
        results.append(predictions)
    return results
