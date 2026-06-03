import logging
import shap
import transformers

logger = logging.getLogger(__name__)

class SentimentExplainer:
    """Original — finBERT has zero explainability"""
    
    def __init__(self, model, tokenizer):
        # Initialize text classification pipeline for SHAP explainer
        self.pipeline = transformers.pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None
        )
        self.explainer = shap.Explainer(self.pipeline)
        
    def _top_k(self, shap_values, tokens, label: int, k: int = 5) -> list:
        if len(shap_values.values) == 0:
            return []
        
        values = shap_values.values[0][:, label]
        data = shap_values.data[0]
        
        token_values = []
        for token, val in zip(data, values):
            if isinstance(token, str):
                clean_token = token.replace("Ġ", "").replace(" ", "").strip()
            else:
                try:
                    tokenizer = getattr(self.explainer, "tokenizer", self.pipeline.tokenizer)
                    clean_token = tokenizer.decode([token]).strip()
                except Exception:
                    clean_token = str(token)
            
            # Filter out special/empty tokens
            if clean_token and not clean_token.startswith("<") and not clean_token.endswith(">") and clean_token not in ("[CLS]", "[SEP]", "[PAD]"):
                token_values.append((clean_token, val))
                
        # Sort descending by SHAP value
        sorted_tokens = sorted(token_values, key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_tokens[:k]]

    def explain(self, text: str) -> dict:
        shap_values = self.explainer([text])
        
        try:
            tokenizer = getattr(self.explainer, "tokenizer", self.pipeline.tokenizer)
            tokens = tokenizer(text)["input_ids"]
        except Exception:
            tokens = []
            
        return {
            "top_bullish_tokens": self._top_k(shap_values, tokens, label=0, k=5),
            "top_bearish_tokens": self._top_k(shap_values, tokens, label=1, k=5),
            "explanation_html": shap.plots.text(shap_values, display=False)
        }
