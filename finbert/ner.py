import logging
import spacy
import spacy.cli

logger = logging.getLogger(__name__)

class EntitySentimentTracker:
    """Completely original — finBERT has no NER layer"""
    
    def __init__(self, sentiment_engine):
        self.sentiment_engine = sentiment_engine
        
        # Load spaCy NLP pipeline (fallback to en_core_web_sm if trf is missing)
        try:
            logger.info("Attempting to load en_core_web_trf...")
            self.nlp = spacy.load("en_core_web_trf")
        except Exception as e:
            logger.warning("Failed to load en_core_web_trf (%s). Falling back to en_core_web_sm...", e)
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as ex:
                logger.warning("en_core_web_sm not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception as dex:
                    logger.error("Failed to download en_core_web_sm: %s", dex)
                    raise dex

    def _score_sentence(self, sent: str) -> dict:
        """Internal helper to score a sentence's sentiment using SentimentEngine."""
        try:
            pred_df = self.sentiment_engine.predict(sent)
            if not pred_df.empty:
                row = pred_df.iloc[0]
                probs = row['logit']
                try:
                    pred_idx = self.sentiment_engine.LABELS.index(row['prediction'])
                    confidence = float(probs[pred_idx])
                except Exception:
                    confidence = 1.0
                return {
                    "prediction": row['prediction'],
                    "sentiment_score": float(row['sentiment_score']),
                    "confidence": confidence
                }
        except Exception as e:
            logger.warning("Error scoring sentence '%s': %s", sent, e)
        return {"prediction": "neutral", "sentiment_score": 0.0, "confidence": 1.0}

    def extract_entity_sentiment(self, text: str, predictions: dict) -> dict:
        """Extract entities of type ORG, GPE, MONEY and score local sentence sentiments."""
        doc = self.nlp(text)
        entity_map = {}
        for ent in doc.ents:
            if ent.label_ in ("ORG", "GPE", "MONEY"):
                # Get sentence containing entity, re-score that sentence
                sent = ent.sent.text
                entity_map[ent.text] = {
                    "entity_type": ent.label_,
                    "local_sentiment": self._score_sentence(sent),
                    "document_sentiment": predictions.get("label") or predictions.get("prediction") or "neutral",
                    "confidence": predictions.get("confidence") or 1.0
                }
        return entity_map
