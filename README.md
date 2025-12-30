CyberBERT: Semantic Log Threat Detection

CyberBERT is a production-grade cybersecurity tool designed to detect malicious intent in system logs and URLs using a fine-tuned DistilBERT architecture. Unlike traditional Regex-based systems, CyberBERT understands the semantics and context of log messages, allowing it to identify obfuscated threats and zero-day patterns.

1. Project Overview

The goal of this project is to move beyond simple keyword matching. We utilize a transformer-based encoder to map log lines into a high-dimensional vector space, classifying them as NORMAL or SUSPICIOUS.

Key Features

Semantic Intelligence: Detects threats like SQL Injection and Privilege Escalation even when obfuscated.

Handling Imbalance: Uses Weighted Cross-Entropy Loss to prioritize high-risk threats over common logs.

Adversarial Robustness: Specifically trained to recognize Base64 payloads and "Goodword" camouflage attacks.

Production Ready: Includes a FastAPI wrapper for real-time inference and MPS support for Apple Silicon acceleration.

2. Technical Workflow

The project is decoupled into four distinct stages:

ETL (real_data_loader.py): Standardizes raw 6-lakh datasets (e.g., Kaggle Malicious URLs) into a balanced 1-lakh training set (50/50 distribution).

Training (trainer.py): Fine-tunes the distilbert-base-uncased model using hardware acceleration (MPS/CUDA).

Deployment (log_threat_detector.py): Serves the fine-tuned model via a REST API.

Validation (testing.py): Runs out-of-distribution and adversarial test cases to verify model generalization.

3. Getting Started

Prerequisites

pip install torch transformers pandas scikit-learn fastapi uvicorn requests


Replication Steps

Prepare Data: Place your raw dataset (e.g., malicious_urls.csv) in the root and run the preprocessor.

python real_data_loader.py


Fine-Tune Model:

python trainer.py


Start API Service:

python log_threat_detector.py


Run Security Tests:

python testing.py


4. Model Performance & Inferences

Our benchmarks show that BERT-based detection significantly outperforms traditional models in:

Generalization: Identifying privilege escalation commands (visudo) it has never seen.

Contextual Awareness: Distinguishing between safe technical "Alert" logs and malicious "Injection" patterns.

Recall Optimization: By weighting the malicious class at 5.0x to 100.0x, the system minimizes False Negatives—the most dangerous failure in security.

5. Future Improvisations

To further scale this for enterprise environments:

Model Quantization: Convert to 8-bit (ONNX/TensorRT) to reduce latency for high-throughput SIEM integration.

Active Learning: Implement a feedback loop where low-confidence predictions are flagged for human SOC analysts and used for re-training.

Contextual Windows: Use Longformer or sliding window approaches for multi-line log sequences to detect "Lateral Movement" across sessions.