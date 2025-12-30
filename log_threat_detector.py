import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import os
from typing import Dict

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogThreatConfig:
    """Production configuration pointing to the local fine-tuned model."""
    # This path must match the save directory in trainer.py
    MODEL_PATH = "./cyber_bert_model" 
    MAX_LEN = 128

class ThreatDetectionModel:
    """Wrapper for the fine-tuned BERT model and production inference logic."""
    def __init__(self, config: LogThreatConfig):
        self.config = config
        
        # Optimized device selection for Mac (MPS), NVIDIA (CUDA), or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Using device: {self.device}")
        
        # Check if the trained model exists
        if not os.path.exists(config.MODEL_PATH):
            logger.error(f"Trained model not found at {config.MODEL_PATH}. Please run trainer.py first.")
            raise FileNotFoundError("Fine-tuned model weights missing.")

        logger.info(f"Loading fine-tuned model from {config.MODEL_PATH}...")
        
        # Load the fine-tuned tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH).to(self.device)
        self.model.eval() # Set to evaluation mode for inference
        
        logger.info("Model loaded successfully.")

    def predict(self, text: str) -> Dict:
        """Production inference logic with confidence scoring."""
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Convert raw logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return {
            "log_line": text,
            "is_threat": bool(prediction),
            "confidence_score": round(confidence, 4),
            "classification": "SUSPICIOUS" if prediction == 1 else "NORMAL",
            "metadata": {
                "model_version": "v1.0-cyber-bert",
                "inference_device": str(self.device)
            }
        }

# --- FastAPI Implementation ---
app = FastAPI(
    title="CyberBERT Real-Time Threat Detection",
    description="BERT-based semantic log analysis for cybersecurity anomaly detection."
)

# Initialize the model once on startup
try:
    detector = ThreatDetectionModel(LogThreatConfig())
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    detector = None

class LogRequest(BaseModel):
    log_line: str

@app.post("/analyze")
async def analyze_log(request: LogRequest):
    """
    Real-time endpoint for security logs.
    Accepts a single log string and returns threat classification.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model service unavailable.")
    
    try:
        result = detector.predict(request.log_line)
        
        # Log high-confidence threats for security alerting
        if result["is_threat"] and result["confidence_score"] > 0.85:
            logger.warning(f"HIGH CONFIDENCE THREAT DETECTED: {request.log_line}")
            
        return result
    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/health")
async def health():
    """Health check for load balancers."""
    return {
        "status": "healthy" if detector else "unhealthy",
        "model_loaded": detector is not None
    }

if __name__ == "__main__":
    # Start the production server
    logger.info("Initializing CyberBERT API Service...")
    uvicorn.run(app, host="127.0.0.1", port=8001)