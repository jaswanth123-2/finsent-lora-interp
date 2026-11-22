# finsent-lora-interp

Fine-tuning Llama-3.2-3B-Instruct with LoRA to classify finance tweets as **Bearish / Bullish / Neutral**, then looking inside the model to see what changed.

**Headline result:** macro-F1 **0.873** on the real validation set (n=2,388), against **0.697** for a TF-IDF + logistic regression baseline. The success criterion (beat 0.697) was written down before training.

Built with Llama.

---

## Results

Validation set, n = 2,388 (Bearish 347, Bullish 475, Neutral 1,566). Macro-F1 is the headline metric because Neutral is 65.6% of the data, so accuracy alone is misleading.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always predict "Neutral" | 0.656 | 0.264 |
| Llama-3.2-3B-Instruct zero-shot (original baseline run) | 0.543 | 0.557 |
| Llama-3.2-3B-Instruct zero-shot (Stage 7 session, adapter disabled) | 0.606 | 0.606 |
| TF-IDF (1-2 grams) + LogReg, trained on the full train split | 0.806 | 0.697 |
| **Llama-3.2-3B-Instruct + LoRA (r=16, lr 2e-4)** | **0.902** | **0.873** |

Fine-tuned model per class:

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

**The fair comparison is TF-IDF (+0.176 macro-F1).** The zero-shot prompt was never optimized, so gains over zero-shot overstate the benefit of fine-tuning.

### About the two zero-shot numbers

The original baseline run scored macro-F1 0.557. Re-running the same prompt later gave about 0.61. I tested three configurations in the later session (with and without double quantization, default dtype, fp32 for non-quantized layers) and all landed between 0.603 and 0.609, so quantization settings and dtype do not explain the gap. The cause is unknown; the likely candidate is a different library environment between sessions (not verified). Zero-shot scores here should be treated as environment-sensitive. The fine-tuned result is unaffected: training and evaluation both used explicit `fp16`.

Versions used for the Stage 7 session: transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2, torch 2.14.1+cu130.

---

## Setup

- **Data:** [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (credit: Ricky Costa). Train 9,543 / validation 2,388. The dataset card has no license filled in, so no license claim is made here.
- **Splits:** 10% stratified split of train (seed 42) gives 8,588 train / 955 dev. The real validation set was never used for training decisions.
- **Model:** `meta-llama/Llama-3.2-3B-Instruct`, 4-bit NF4 with double quantization, LoRA via PEFT.
- **LoRA:** r=16, alpha=32, dropout 0.05, on all 7 linear modules (q, k, v, o, gate, up, down). 24,313,856 trainable parameters of 3,237,063,680 (0.7511%).
- **Training:** loss on answer tokens only; batch 8 x grad-accum 2 (effective 16); lr 2e-4; 2 epochs (1,074 steps); cosine schedule, 30 warmup steps; fp16; gradient checkpointing; seed 42. About 1 hour on a free Colab T4.
- **Evaluation:** compare the next-token scores of the first token of each label word. Note that the first tokens are `Bear`, `B` and `Neutral`, so Bullish is scored on the bare token `B`. The three ids are distinct, so the method works, but `B` is a very generic token.
- **Pinned prompt date:** Llama's chat template inserts today's date unless pinned. All runs use `01 Jan 2026`.

---

## Ablation: LoRA rank 8 vs 16

One variable changed (rank 8, alpha 16 so alpha/r stays 2). Compared on the 955-example dev split, not on validation.

| | Trainable params | Dev accuracy | Dev macro-F1 |
|---|---|---|---|
| r = 8 | about 12 million | 0.9152 | 0.8876 |
| r = 16 (main model) | about 24 million | 0.9225 | 0.8992 |

Halving the rank cost about 1 point of macro-F1. The gap is about 7 tweets out of 955, from a single run with one seed, so it may be noise. This is reported as an ablation, not tuning; the main model stays r=16 as pre-registered.

---

## Looking inside the model

Three checks on the main model. All are correlational, from one model, one task and one seed.

### (a) Layer drift

Cosine similarity of the final-token hidden state, base vs fine-tuned, over 500 validation tweets.

- Layers 0-4 barely change (similarity above 0.99).
- Drift grows fast through layers 5-11 and is largest at layer 13 (similarity 0.69).
- It partly recovers toward the last layer (0.82).
- The spread across tweets grows after layer 13 (worst tweet about 0.49 at layer 17), so some tweets change much more than others. Which ones was not analyzed.

Figure: `results/fig_layer_drift.png`

### (b) Linear probes

A logistic regression per layer, fit on 2,000 training-tweet hidden states and tested on 1,000 validation tweets.

| Layer | Base macro-F1 | Fine-tuned macro-F1 |
|---|---|---|
| 5 | 0.677 | 0.702 |
| 8 | 0.731 | 0.797 |
| 10 | 0.755 | 0.855 |
| 15 | 0.852 (base peak) | 0.881 |
| 19 | 0.841 | 0.892 (fine-tuned peak) |
| 28 | 0.840 | 0.887 |

- The base model already encodes sentiment: a probe reads it at macro-F1 0.85 from layer 15, even though its zero-shot answers score about 0.61. The information was there; the zero-shot output did not use it well.
- Fine-tuning moved it earlier (macro-F1 0.8 is reached at layer 9 instead of 13; the biggest gaps are in layers 8-12) and raised the ceiling by about 0.04.
- Sanity check: at layer 0 both models score 0.2634, which equals always-Neutral. The last token is identical in every prompt, so nothing leaks into the probe.

Figure: `results/fig_probes.png`

### (c) LoRA update size

Frobenius norm of (alpha/r) * B * A per layer and module.

- By module (mean over layers): gate_proj 1.64, up_proj 1.31, q_proj 0.91, down_proj 0.84, o_proj 0.71, k_proj 0.55, v_proj 0.45. The learned change is concentrated in the MLP.
- By depth: update sizes are largest in early layers (layer 1 totals about 8.9) and shrink toward the end (layer 27 about 3.7).

Figure: `results/fig_lora_norms.png`

### An unexpected result

The LoRA updates are biggest in the early layers, but the hidden states barely change there (layers 0-4). The hidden states change most in the middle layers (11-17). A large weight update therefore did not mean a large change in activations. A possible explanation is that small changes accumulate through the layers, but this was not tested.

---

## Error analysis

The fine-tuned model makes 234 errors out of 2,388 (9.8%).

| Error | Count | Share |
|---|---|---|
| Bullish -> Neutral | 128 | 55% |
| Bearish -> Neutral | 57 | 24% |
| Neutral -> Bearish | 35 | 15% |
| Neutral -> Bullish | 13 | 6% |
| Bullish -> Bearish | 1 | 0.4% |

- 79% of errors are a Bullish or Bearish tweet predicted Neutral. The model rarely gets the direction wrong (1 error).
- The main failure of the zero-shot baseline, Neutral -> Bearish, dropped from 735 cases to 35.
- Compared on the same tweets, fine-tuning fixed about 858 tweets and broke 151. The 151 are largely a side effect of the model becoming more conservative: the base model over-called Bullish and Bearish.
- Errors are not explained by tweet length (12.7 words in errors, 12.3 in correct tweets), by links, or by any single keyword; the word-count differences were small and the counts tiny.
- From reading a sample of tweets (not a measured result), many Bullish -> Neutral errors are plain facts that need market knowledge to read as positive, and some Neutral -> Bearish errors look arguably correct. Whether this reflects label disagreement was **not verified**; it would need a manual audit.

---

## Honest limits

- One small model, one task, one dataset, one seed per configuration. No confidence intervals.
- The zero-shot prompt was never optimized; TF-IDF is the fair baseline.
- The zero-shot number was not reproducible across Colab sessions (0.557 vs about 0.61); cause unknown.
- The rank 8 vs 16 difference is small and may be noise.
- Interpretability checks are correlational. Probe numbers use a 1,000-tweet sample, so differences under about 0.02 are within noise. Several probe fits reported sklearn convergence warnings (`max_iter=300`), which may shift exact values slightly (both models treated the same way). Probe scores are not comparable to the headline 0.873.
- Label quality in the dataset was not audited.

---

## Repository layout

```
notebooks/
  1_baseline.ipynb            baselines before training
  2_train_lora_r16.ipynb      main LoRA training
  3_final_eval.ipynb          final evaluation on validation
  4_ablation_rank8.ipynb      rank 8 vs 16 ablation (dev split)
  5_interpretability.ipynb    layer drift, probes, LoRA update size
  6_error_analysis.ipynb      error analysis
results/
  baseline_zero_shot.json     original zero-shot baseline run
  eval_finetuned_r16_lr2e-4.json, eval_base_r16_lr2e-4.json
  ablation_dev_r8_vs_r16.json
  log_r16_lr2e-4.json, log_r8_lr2e-4.json    training logs
  layer_drift.csv, probe_results.csv, lora_update_norms.csv, interp_summary.json
  ft_errors.csv
  fig_layer_drift.png, fig_probes.png, fig_lora_norms.png
```

Adapters are not in this repo. They are on the Hugging Face Hub: **[YOUR-HF-USERNAME]/[ADAPTER-REPO-NAME]** (add the link after Stage 10).

## License and attribution

- Data: Ricky Costa, `zeroshot/twitter-financial-news-sentiment` (dataset card lists no license).
- Base model: Llama 3.2, used under the Llama 3.2 Community License. Built with Llama.
- Code in this repository: add your own license choice here (for example MIT).
