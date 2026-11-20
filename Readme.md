# FinSent LoRA + Interpretability

Fine-tune Llama-3.2-3B-Instruct with LoRA to classify finance tweets 
as Bearish / Bullish / Neutral, then analyse what changed inside the 
model (layer drift, linear probes, LoRA update size per layer).

Dataset: zeroshot/twitter-financial-news-sentiment (credit: Ricky Costa)

## Status: Stage 1 - setup