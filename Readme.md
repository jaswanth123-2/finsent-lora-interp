# finsent-lora-interp

LoRA fine-tuning of Llama-3.2-3B-Instruct for financial tweet sentiment, with a layer-wise analysis of what fine-tuning changes.

Built with Llama. Adapter on Hugging Face: [Jaswanth10/Llama-3.2-3B-FinSent-LoRA](https://huggingface.co/Jaswanth10/Llama-3.2-3B-FinSent-LoRA)

## Summary

- **Task:** classify a finance tweet as Bearish, Bullish or Neutral.
- **Method:** LoRA (rank 16) on a 4-bit Llama-3.2-3B-Instruct. 24.3M trainable parameters, 0.75% of the model.
- **Result:** macro-F1 **0.873** on the held-out validation set (n=2,388).
- **Baseline to beat:** TF-IDF + logistic regression, macro-F1 0.697 (fixed before training).
- **Ablation:** rank 8 is about 1 point of macro-F1 below rank 16.
- **Analysis:** the base model already encodes sentiment inside; fine-tuning moves it to earlier layers.

## Data

- **Dataset:** [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (credit: Ricky Costa).
- **Labels:** 0 = Bearish, 1 = Bullish, 2 = Neutral.
- **Size:** train 9,543 tweets, validation 2,388 tweets.
- **Class balance (validation):** Neutral 1,566 (65.6%), Bullish 475, Bearish 347.
- **Split used:**
  - Train: 8,588 tweets (the model learns from these).
  - Dev: 955 tweets (10% of train, stratified, seed 42; used for quick checks and the ablation).
  - Validation: 2,388 tweets (final test only, never used for any decision).

## Method

1. **Metric:** macro-F1 is the headline score. A model that always says Neutral gets 0.656 accuracy, so accuracy alone is misleading.
2. **Success rule (set before training):** fine-tuned macro-F1 must beat 0.697.
3. **Prompt:** one fixed chat prompt for every run. The date is pinned to `01 Jan 2026`, because the Llama template otherwise inserts today's date.
4. **Prediction:** take the model's next-token scores for the first token of each label word and pick the highest.
   - The first tokens are `Bear`, `B` and `Neutral`, so Bullish is read from the generic token `B`.
5. **Model loading:** `meta-llama/Llama-3.2-3B-Instruct` in 4-bit NF4 with double quantization, fp16 compute.
6. **LoRA setup:**
   - Rank 16, alpha 32, dropout 0.05.
   - Applied to q, k, v, o, gate, up and down projections in all 28 layers.
   - 24,313,856 of 3,237,063,680 parameters are trainable (0.7511%). The rest are frozen.
7. **Training:**
   - Loss on the answer tokens only.
   - Batch 8 with 2 accumulation steps (effective 16).
   - Learning rate 2e-4, cosine schedule, 30 warmup steps.
   - 2 epochs (1,074 steps), fp16, gradient checkpointing, seed 42.
   - About 1 hour on a free Colab T4.
8. **Evaluation:** score the 2,388 validation tweets once, with the adapter on and with it switched off (base model), on the same loaded model.

## Results

Validation set, n = 2,388.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always "Neutral" | 0.656 | 0.264 |
| Llama-3.2-3B zero-shot (original run) | 0.543 | 0.557 |
| Llama-3.2-3B zero-shot (later session, adapter off) | 0.606 | 0.606 |
| TF-IDF (1-2 grams) + LogReg | 0.806 | 0.697 |
| **Llama-3.2-3B + LoRA (r=16)** | **0.902** | **0.873** |

- The success rule is met: 0.873 is +0.176 above TF-IDF.
- TF-IDF is the fair baseline, because the zero-shot prompt was never optimized.

Per class (fine-tuned):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bearish | 0.89 | 0.84 | 0.86 |
| Bullish | 0.96 | 0.73 | 0.83 |
| Neutral | 0.89 | 0.97 | 0.93 |

Confusion matrix (rows = true, columns = predicted; Bearish, Bullish, Neutral):

```
[[ 290,   0,   57],
 [   1, 346,  128],
 [  35,  13, 1518]]
```

- The model is cautious: precise on Bullish and Bearish, with lower recall.
- Neutral predicted as Bearish, the main zero-shot problem, fell from 735 cases to 35.

## Ablation: rank 8 vs rank 16

- **What changed:** only the rank (8 instead of 16, alpha 16 so alpha/r stays 2).
- **Where compared:** the 955-tweet dev split, so the validation set stays untouched.

| Rank | Trainable params | Dev accuracy | Dev macro-F1 |
|---|---|---|---|
| 8 | 12.2M | 0.9152 | 0.8876 |
| 16 | 24.3M | 0.9225 | 0.8992 |

- Half the parameters cost about 1 point of macro-F1 (about 7 of 955 tweets).
- This is one run per rank, so the gap could be noise. I do not claim rank 16 is better.

## Layer-wise analysis

**Setup**
1. Take the final prompt token's hidden state at 29 points: the input plus each of the 28 layers.
2. Do this with the adapter on and with it off.
3. Run three analyses on these states.

**(a) Drift: how different are the two models at each layer?**
- Measure: cosine similarity, 500 validation tweets. 1.0 means identical.
- Layer 0 is 1.000, as expected (sanity check).
- Layers 0-4 stay above 0.99.
- Similarity drops through layers 5-11, with the minimum at layer 13 (0.69).
- It partly recovers to 0.82 at the last layer.

**(b) Probes: where can sentiment be read from the hidden state?**
- Measure: a logistic regression per layer, fit on 2,000 training tweets, tested on 1,000 validation tweets.

| Layer | Base macro-F1 | Fine-tuned macro-F1 |
|---|---|---|
| 0 | 0.263 | 0.263 |
| 5 | 0.677 | 0.702 |
| 8 | 0.731 | 0.797 |
| 10 | 0.755 | 0.855 |
| 15 | 0.852 (base peak) | 0.881 |
| 19 | 0.841 | 0.892 (fine-tuned peak) |
| 28 | 0.840 | 0.887 |

- Layer 0 equals always-Neutral for both, as it should (the last token is identical in every prompt).
- The base model already encodes sentiment (0.85 at layer 15), though its zero-shot answers score about 0.61.
- Fine-tuning moves it earlier: macro-F1 0.8 is reached at layer 9 instead of 13.
- Fine-tuning also raises the ceiling by about 0.04.

**(c) Update size: where is the learned correction large?**
- Measure: norm of the LoRA update, (alpha/r)·B·A, per layer and module.
- By module (mean over layers): gate 1.64, up 1.31, q 0.91, down 0.84, o 0.71, k 0.55, v 0.45.
- By depth: largest in early layers (layer 1 about 8.9), smaller toward the end (layer 27 about 3.7).

**Observation**
- The biggest weight updates are in early layers, but the hidden states barely change there.
- Hidden states change most in the middle layers.
- Possible reason: small early changes build up as they pass through the layers. This was not tested.

Figures: `results/fig_layer_drift.png`, `results/fig_probes.png`, `results/fig_lora_norms.png`

## Limitations

- One model, one task, one dataset.
- One seed per run, no confidence intervals.
- The rank 8 vs 16 gap may be noise.
- The zero-shot prompt was never optimized.
- Zero-shot score changed between sessions (0.557 to about 0.61). The cause is unknown.
- Layer analysis shows what changed, not why.
- Probe gaps below about 0.02 are noise.
- Some probe fits did not fully converge.
- Dataset labels were not checked for errors.

## Repository

```
notebooks/
  1_baseline.ipynb            baselines before training
  2_train_lora_r16.ipynb      LoRA training
  3_final_eval.ipynb          validation evaluation
  4_ablation_rank8.ipynb      rank 8 vs 16 (dev split)
  5_interpretability.ipynb    drift, probes, update size
results/
  baseline_zero_shot.json
  eval_finetuned_r16_lr2e-4.json, eval_base_r16_lr2e-4.json
  ablation_dev_r8_vs_r16.json
  log_r16_lr2e-4.json, log_r8_lr2e-4.json
  layer_drift.csv, probe_results.csv, lora_update_norms.csv, interp_summary.json
  fig_layer_drift.png, fig_probes.png, fig_lora_norms.png
```

## License and credit

- Code: MIT License (see `LICENSE`).
- Base model: Llama 3.2 Community License. Built with Llama.
- Data: Ricky Costa. The dataset card lists no license, so none is claimed here.