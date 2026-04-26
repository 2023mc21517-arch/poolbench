## Leaderboard Submission

### Model / strategy being evaluated

| Field | Value |
|---|---|
| **Model name** | <!-- e.g. Llama-3.1-70B, GPT-4o --> |
| **Model HF ID** | <!-- e.g. meta-llama/Meta-Llama-3.1-70B --> |
| **Strategy ID(s)** | <!-- e.g. A1, P2 — use IDs from the pooling strategies table --> |
| **Construction method** | <!-- C1 / C2 / C3 / C4 / C5 --> |

### Results summary

Paste the headline numbers from your `results/auroc/` JSON:

| Concept | AUROC | 95% CI |
|---|---|---|
| hedging | | |
| legal_formality | | |
| math_certainty | | |
| (add rows...) | | |

**Mean AUROC across all 18 concepts:** ___

### Files added

- [ ] `results/auroc/<model_id>.json`
- [ ] `results/scp/<model_id>.json` *(if available)*
- [ ] `results/disentanglement/<model_id>.json` *(if available)*

### Checklist

- [ ] I ran `python scripts/dataset_builder.py --all` from scratch (no modified corpus files)
- [ ] I ran `python scripts/power_analysis.py` and it passed the 95% CI < 0.025 criterion
- [ ] No corpus JSONL files, activation `.npy` files, or model weights are included in this PR
- [ ] No HuggingFace tokens or API keys are included
- [ ] Results JSON schema matches the existing files in `results/`
- [ ] I have read and agree to the [Contributing guidelines](../CONTRIBUTING.md)

### Reproduction command

```bash
python scripts/run_model.py --model <your_model_id> --device cuda:0
```

### Notes

<!-- Any caveats, hardware used, approximations made, or differences from the standard pipeline. -->
