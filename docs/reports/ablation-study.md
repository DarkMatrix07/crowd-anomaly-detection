# Ablation Study

## Scope

Synthetic placeholder run for pipeline verification. Replace with real ShanghaiTech scores after dataset training.

## Results

| variant | roc_auc | pr_auc | f1 | event_f1 | false_alert_rate_per_10m |
| --- | --- | --- | --- | --- | --- |
| alternate_backbone | 1.0 | 1.0 | 0.9969 | 0.8 | 2.5 |
| baseline | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| no_density | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| no_flow | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| no_smoothing | 1.0 | 1.0 | 0.9938 | 0.6667 | 5.0 |

## Recommended Variant

- Best tradeoff variant: `baseline`
- Tradeoff score: `1.0000`

## Next Action

Re-run this script with real model outputs after dataset training and supervisor-reviewed thresholds.