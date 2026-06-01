# Adversarial Bot Detector (Trust & Safety)

This repository demonstrates a production-minded approach to catching adversarial behavior and platform abuse using Sequence Modeling.

In Trust & Safety, static rules fail against coordinated bot rings. This pipeline uses a **PyTorch Transformer Encoder** to analyze user action sequences (e.g., logins, likes, comments, API requests) and classify them as benign or adversarial based on temporal and behavioral patterns.

## Architecture
1. **Sequence Generation:** Simulates temporal user logs (time between actions, action types).
2. **Transformer Encoder:** Captures the complex sequential relationships in user behavior that traditional ML models miss.
3. **Fraud Classification:** Outputs a probability score, flagging anomalous, high-velocity bursts typical of adversarial attacks.
