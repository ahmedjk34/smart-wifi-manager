# Step 4A Model Benchmarking Results

Generated on: 2025-08-19 18:53:56
Total models evaluated: 6

## Summary

| Model | Label | Accuracy |
|-------|-------|----------|
| rf_oracle | oracle_best_rateIdx | 0.9984 |
| lgb_oracle | oracle_best_rateIdx | 0.9155 |
| xgb_oracle | oracle_best_rateIdx | 0.9987 |
| rf_v3 | v3_rateIdx | 0.9542 |
| lgb_v3 | v3_rateIdx | 0.8581 |
| xgb_v3 | v3_rateIdx | 0.9670 |

## Model: rf_oracle | Label: oracle_best_rateIdx
- Accuracy: 0.9984
### Per-class metrics:
  - Class 0: P=1.000, R=0.800, F1=0.889
  - Class 1: P=0.710, R=0.835, F1=0.767
  - Class 2: P=0.880, R=0.458, F1=0.603
  - Class 3: P=0.551, R=0.607, F1=0.578
  - Class 4: P=0.775, R=0.924, F1=0.843
  - Class 5: P=0.757, R=0.582, F1=0.658
  - Class 6: P=1.000, R=0.156, F1=0.270
  - Class 7: P=1.000, R=0.999, F1=0.999
  - Class 8: P=1.000, R=1.000, F1=1.000
  - Class 9: P=1.000, R=1.000, F1=1.000
  - Class 10: P=0.992, R=0.969, F1=0.980
  - Class 11: P=1.000, R=1.000, F1=1.000
  - Class macro avg: P=0.889, R=0.778, F1=0.799
  - Class weighted avg: P=0.999, R=0.998, F1=0.998

## Model: lgb_oracle | Label: oracle_best_rateIdx
- Accuracy: 0.9155
### Per-class metrics:
  - Class 0: P=0.000, R=0.000, F1=0.000
  - Class 1: P=0.000, R=0.000, F1=0.000
  - Class 2: P=0.000, R=0.000, F1=0.000
  - Class 3: P=0.000, R=0.000, F1=0.000
  - Class 4: P=0.000, R=0.000, F1=0.000
  - Class 5: P=0.000, R=0.000, F1=0.000
  - Class 6: P=0.000, R=0.000, F1=0.000
  - Class 7: P=0.206, R=0.191, F1=0.199
  - Class 8: P=0.012, R=0.013, F1=0.012
  - Class 9: P=0.409, R=0.294, F1=0.342
  - Class 10: P=0.088, R=0.028, F1=0.042
  - Class 11: P=0.952, R=0.966, F1=0.959
  - Class macro avg: P=0.139, R=0.124, F1=0.130
  - Class weighted avg: P=0.905, R=0.916, F1=0.910

## Model: xgb_oracle | Label: oracle_best_rateIdx
- Accuracy: 0.9987
### Per-class metrics:
  - Class 0: P=1.000, R=0.800, F1=0.889
  - Class 1: P=0.789, R=0.899, F1=0.840
  - Class 2: P=0.717, R=0.688, F1=0.702
  - Class 3: P=0.596, R=0.757, F1=0.667
  - Class 4: P=0.895, R=0.890, F1=0.893
  - Class 5: P=0.667, R=0.637, F1=0.652
  - Class 6: P=0.915, R=0.558, F1=0.694
  - Class 7: P=0.999, R=0.999, F1=0.999
  - Class 8: P=0.994, R=0.994, F1=0.994
  - Class 9: P=0.995, R=0.994, F1=0.995
  - Class 10: P=0.983, R=0.993, F1=0.988
  - Class 11: P=1.000, R=1.000, F1=1.000
  - Class macro avg: P=0.879, R=0.851, F1=0.859
  - Class weighted avg: P=0.999, R=0.999, F1=0.999

## Model: rf_v3 | Label: v3_rateIdx
- Accuracy: 0.9542
### Per-class metrics:
  - Class 0: P=0.997, R=0.993, F1=0.995
  - Class 1: P=0.754, R=0.736, F1=0.745
  - Class 2: P=0.875, R=0.592, F1=0.706
  - Class 3: P=0.898, R=0.602, F1=0.721
  - Class 4: P=0.698, R=0.622, F1=0.658
  - Class 5: P=0.817, R=0.763, F1=0.789
  - Class 6: P=0.868, R=0.693, F1=0.771
  - Class 7: P=0.971, R=0.692, F1=0.808
  - Class 8: P=0.942, R=0.748, F1=0.834
  - Class 9: P=0.955, R=0.998, F1=0.976
  - Class 10: P=0.780, R=0.669, F1=0.720
  - Class 11: P=0.998, R=0.998, F1=0.998
  - Class macro avg: P=0.880, R=0.759, F1=0.810
  - Class weighted avg: P=0.953, R=0.954, F1=0.951

## Model: lgb_v3 | Label: v3_rateIdx
- Accuracy: 0.8581
### Per-class metrics:
  - Class 0: P=0.889, R=0.948, F1=0.917
  - Class 1: P=0.214, R=0.347, F1=0.265
  - Class 2: P=0.092, R=0.148, F1=0.114
  - Class 3: P=0.446, R=0.376, F1=0.408
  - Class 4: P=0.109, R=0.335, F1=0.164
  - Class 5: P=0.561, R=0.400, F1=0.467
  - Class 6: P=0.771, R=0.700, F1=0.734
  - Class 7: P=0.726, R=0.547, F1=0.624
  - Class 8: P=0.743, R=0.598, F1=0.662
  - Class 9: P=0.920, R=0.937, F1=0.929
  - Class 10: P=0.037, R=0.247, F1=0.065
  - Class 11: P=0.727, R=0.703, F1=0.715
  - Class macro avg: P=0.520, R=0.524, F1=0.505
  - Class weighted avg: P=0.865, R=0.858, F1=0.860

## Model: xgb_v3 | Label: v3_rateIdx
- Accuracy: 0.9670
### Per-class metrics:
  - Class 0: P=0.997, R=0.993, F1=0.995
  - Class 1: P=0.814, R=0.868, F1=0.840
  - Class 2: P=0.825, R=0.732, F1=0.776
  - Class 3: P=0.919, R=0.926, F1=0.923
  - Class 4: P=0.748, R=0.834, F1=0.789
  - Class 5: P=0.903, R=0.863, F1=0.882
  - Class 6: P=0.919, R=0.720, F1=0.808
  - Class 7: P=0.978, R=0.875, F1=0.924
  - Class 8: P=0.940, R=0.819, F1=0.876
  - Class 9: P=0.967, R=0.998, F1=0.982
  - Class 10: P=0.971, R=0.899, F1=0.933
  - Class 11: P=1.000, R=1.000, F1=1.000
  - Class macro avg: P=0.915, R=0.877, F1=0.894
  - Class weighted avg: P=0.966, R=0.967, F1=0.965

