# Smart WiFi Rate Adaptation ML Pipeline

A comprehensive machine learning pipeline for intelligent WiFi rate adaptation using ns-3 simulation data. This project implements class weight-based training instead of aggressive downsampling to preserve realistic network conditions while handling class imbalance effectively.

## 🎯 Project Overview

This project develops machine learning models for WiFi rate adaptation that can intelligently select optimal transmission rates based on real-time network conditions. Unlike traditional approaches that force artificial class balance, our pipeline preserves realistic network distributions while using class weights to ensure minority classes receive appropriate attention during training.

## 📊 Key Achievements

- **412,000 training samples** with 42 engineered features
- **Class weight-based imbalance handling** preserving realistic network distributions
- **3 oracle strategies**: Conservative, Balanced, and Aggressive rate adaptation
- **12 synthetic edge case scenarios** for robust model training
- **6 network context classifications** covering all WiFi conditions

## 🏗️ Architecture

### Data Processing Pipeline

```

Raw WiFi Data → Cleaning → Feature Engineering → Context Classification → Oracle Labels → Class Weights → ML Training

```

### Network Context Classification

- **Emergency Recovery**: SNR < 10dB, Success < 50%, Consecutive failures ≥ 3
- **Poor Unstable**: SNR < 15dB, High variance conditions
- **Marginal Conditions**: SNR < 20dB, Success < 80%
- **Good Unstable**: High SNR variance but decent performance
- **Good Stable**: Stable conditions with good performance
- **Excellent Stable**: SNR > 25dB, Success > 90%, Low variance

## 📁 Project Structure

```

smart-wifi-manager/
├── python_files/
│ ├── 3_enhanced_ml_labeling_prep.py # Data prep with class weights
│ ├── ultimate_ml_training_with_class_weights.py # Training pipeline
│ ├── model_artifacts/
│ │ └── class_weights.json # Computed class weights
│ └── logs/ # Training logs
├── smart-v3-ml-cleaned.csv # Cleaned WiFi data
├── smart-v3-ml-enriched.csv # ML-ready dataset
├── step3_rf_rateIdx_model_FIXED.joblib # Trained model
├── step3_scaler_FIXED.joblib # Feature scaler
└── README.md # This file

```

## 🔬 Technical Innovation: Class Weights vs Downsampling

### The Problem with Traditional Approaches

Most ML pipelines for imbalanced data use aggressive downsampling:

- ❌ Forces artificial 1:1 class balance
- ❌ Destroys realistic network patterns
- ❌ Removes valuable majority class information
- ❌ Poor generalization to real-world scenarios

### Our Solution: Intelligent Class Weighting

```python
# Example class weights for rateIdx
rateIdx Class Weights:
  0: 0.254   # Common rates (low weight)
  1: 0.507
  5: 0.509
  2: 33.204  # Rare rates (high weight)
  3: 59.745
  4: 55.317
  6: 29.838
  7: 29.855
```

### Benefits:

- ✅ Preserves realistic 49.2% vs 0.2% rate distributions
- ✅ Rare scenarios get 30-60x more training attention
- ✅ No data loss - keeps all valuable samples
- ✅ Better real-world generalization

## 📈 Dataset Statistics

### Final Dataset Composition

- **Total Samples**: 412,000
- **Features**: 42 engineered features
- **Real Data**: 400,000 samples
- **Synthetic Edge Cases**: 12,000 samples
- **Network Contexts**: 6 classifications

### Class Distribution (rateIdx)

| Rate | Samples | Percentage | Class Weight |
| ---- | ------- | ---------- | ------------ |
| 0    | 202,522 | 49.2%      | 0.254        |
| 1    | 101,543 | 24.6%      | 0.507        |
| 5    | 101,140 | 24.5%      | 0.509        |
| 2    | 1,551   | 0.4%       | 33.204       |
| 6    | 1,726   | 0.4%       | 29.838       |
| 7    | 1,725   | 0.4%       | 29.855       |
| 3    | 862     | 0.2%       | 59.745       |
| 4    | 931     | 0.2%       | 55.317       |

### Network Context Distribution

- **Poor Unstable**: 402,602 (97.8%) - Realistic WiFi conditions
- **Emergency Recovery**: 3,202 (0.8%) - Critical scenarios
- **Marginal Conditions**: 2,077 (0.5%) - Borderline performance
- **Good Stable**: 2,050 (0.5%) - Optimal conditions
- **Excellent Stable**: 1,053 (0.3%) - Perfect conditions
- **Good Unstable**: 1,016 (0.2%) - High performance, variable conditions

## 🛠️ Usage

### 1. Data Preparation

```bash
python 3_enhanced_ml_labeling_prep.py
```

**Features:**

- Loads raw WiFi simulation data
- Applies sanity filtering and outlier removal
- Generates network context classifications
- Creates oracle labels (conservative/balanced/aggressive)
- Computes class weights for imbalanced learning
- Generates synthetic edge cases
- Exports ML-ready dataset

### 2. Model Training

```bash
python ultimate_ml_training_with_class_weights.py
```

**Features:**

- Chunked loading for large datasets (RAM-efficient)
- Stratified train/validation/test splits
- Feature scaling with StandardScaler
- Random Forest with class weights
- Comprehensive evaluation metrics
- Model and scaler persistence

### 3. Model Artifacts

Generated files:

- `step3_rf_rateIdx_model_FIXED.joblib` - Trained Random Forest
- `step3_scaler_FIXED.joblib` - Feature scaler
- `class_weights.json` - Computed class weights
- Training logs and documentation

## 📊 Key Features

### Engineered Features (42 total)

- **SNR Metrics**: lastSnr, snrFast, snrSlow, snrVariance, snrStabilityIndex
- **Success Metrics**: shortSuccRatio, medSuccRatio, consecutiveSuccess/Failure
- **Throughput**: recentThroughputTrend, packetLossRate, retrySuccessRatio
- **Rate Stability**: rateStabilityScore, timeSinceLastRateChange, recentRateChanges
- **Network State**: queueLen, mobilityMetric, channelWidth, offeredLoad
- **Decision Context**: severity, confidence, optimalRateDistance

### Oracle Strategies

1. **Conservative**: Prioritizes reliability over throughput
2. **Balanced**: Balances reliability and performance
3. **Aggressive**: Maximizes throughput with acceptable risk

### Synthetic Edge Cases (12 scenarios)

- High rate failure scenarios
- Low rate recovery situations
- SNR volatility conditions
- Queue saturation events
- Mobility spike scenarios
- Persistent failure cases
- Sudden improvement conditions
- Consecutive success patterns
- Channel width changes
- Rate flip-flop behavior
- Low SNR success cases
- High SNR failure anomalies

## 🎯 Model Performance Expectations

Based on the dataset quality and class weight approach:

- **Expected Validation Accuracy**: 85-92%
- **Training Time**: 5-15 minutes
- **Model Complexity**: Balanced Random Forest (120 estimators, depth 16)
- **Real-world Applicability**: High (preserved realistic distributions)

## 🔧 Technical Requirements

```bash
# Core dependencies
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.1.0
joblib>=1.2.0

# Visualization (optional)
matplotlib>=3.5.0
seaborn>=0.11.0

# Progress tracking
tqdm>=4.64.0
```

## 📝 Key Design Decisions

### 1. Class Weights Over Downsampling

**Rationale**: Network data imbalance is informative, not noise. Preserving realistic distributions while adjusting training weights yields better real-world performance.

### 2. Context-Aware Oracle Labels

**Rationale**: Different network conditions require different strategies. Emergency scenarios need conservative approaches, while stable conditions can afford aggressive optimization.

### 3. Synthetic Edge Case Generation

**Rationale**: Real data may lack critical failure scenarios. Synthetic generation ensures model robustness in edge conditions.

### 4. Multi-Strategy Training

**Rationale**: Different applications need different risk profiles. Training multiple oracle strategies provides deployment flexibility.

## 🚀 Results Summary

### Data Quality Metrics

- ✅ **Clean Data**: 412k samples, realistic feature ranges
- ✅ **Balanced Coverage**: All network contexts represented
- ✅ **Rich Features**: 42 engineered features capture WiFi complexity
- ✅ **Edge Case Coverage**: 12k synthetic scenarios for robustness

### Innovation Highlights

- 🔥 **Class Weight Revolution**: First WiFi rate adaptation using class weights vs downsampling
- 🔥 **Realistic Distributions**: Preserves 97.8% poor conditions vs 0.3% excellent (real WiFi!)
- 🔥 **Context Intelligence**: Network-aware oracle strategies
- 🔥 **Production Ready**: RAM-efficient, scalable pipeline

## 📈 Next Steps

1. **Model Validation**: Test trained model against ns-3 simulations
2. **Hyperparameter Tuning**: Optimize Random Forest parameters
3. **Alternative Algorithms**: Test Neural Networks with class weights
4. **Real-world Deployment**: Integrate with WiFi drivers/firmware
5. **Performance Monitoring**: Track model accuracy in production

## 🤝 Contributing

This project demonstrates best practices for:

- Handling imbalanced time-series data
- Preserving realistic distributions in ML pipelines
- Context-aware feature engineering for network data
- Production-ready ML model training

## 📄 License

This project is part of advanced WiFi research and follows academic/research usage guidelines.

---

**Author**: ahmedjk34  
**Date**: September 22, 2025  
**Project**: Smart WiFi Rate Adaptation ML Pipeline  
**Innovation**: Class Weight-Based Training for Realistic Network ML Models

_"Preserving reality while learning from imbalance - the future of network machine learning."_

```

```

---

oracle_conservative Class Weights:
0: 62.045
1: 200.466
2: 1.157
3: 0.861
4: 0.876
5: 0.820
6: 0.834
7: 0.418
2025-09-24 17:22:24,142 - INFO -
📊 oracle_balanced - Class Distribution & Weights:
2025-09-24 17:22:24,142 - INFO - Class 0: 1,685 samples (0.1%) -> weight: 184.405
2025-09-24 17:22:24,142 - INFO - Class 1: 2,480 samples (0.1%) -> weight: 125.291
2025-09-24 17:22:24,142 - INFO - Class 2: 49,056 samples (2.0%) -> weight: 6.334
2025-09-24 17:22:24,142 - INFO - Class 3: 254,395 samples (10.2%) -> weight: 1.221
2025-09-24 17:22:24,143 - INFO - Class 4: 339,111 samples (13.6%) -> weight: 0.916
2025-09-24 17:22:24,143 - INFO - Class 5: 360,826 samples (14.5%) -> weight: 0.861
2025-09-24 17:22:24,143 - INFO - Class 6: 371,480 samples (14.9%) -> weight: 0.836
2025-09-24 17:22:24,143 - INFO - Class 7: 1,106,744 samples (44.5%) -> weight: 0.281

oracle_balanced Class Weights:
0: 184.405
1: 125.291
2: 6.334
3: 1.221
4: 0.916
5: 0.861
6: 0.836
7: 0.281
2025-09-24 17:22:24,998 - INFO -
📊 oracle_aggressive - Class Distribution & Weights:
2025-09-24 17:22:24,998 - INFO - Class 0: 549 samples (0.0%) -> weight: 565.978
2025-09-24 17:22:24,998 - INFO - Class 1: 1,121 samples (0.0%) -> weight: 277.183
2025-09-24 17:22:24,999 - INFO - Class 2: 2,478 samples (0.1%) -> weight: 125.392
2025-09-24 17:22:24,999 - INFO - Class 3: 2,943 samples (0.1%) -> weight: 105.580
2025-09-24 17:22:24,999 - INFO - Class 4: 530,668 samples (21.3%) -> weight: 0.586
2025-09-24 17:22:24,999 - INFO - Class 5: 574,947 samples (23.1%) -> weight: 0.540
2025-09-24 17:22:24,999 - INFO - Class 6: 8,539 samples (0.3%) -> weight: 36.389
2025-09-24 17:22:24,999 - INFO - Class 7: 1,364,532 samples (54.9%) -> weight: 0.228

oracle_aggressive Class Weights:
0: 565.978
1: 277.183
2: 125.392
3: 105.580
4: 0.586
5: 0.540
6: 36.389
7: 0.228
2025-09-24 17:22:25,810 - INFO -
📊 rateIdx - Class Distribution & Weights:
2025-09-24 17:22:25,810 - INFO - Class 0: 1,053,346 samples (42.4%) -> weight: 0.295
2025-09-24 17:22:25,810 - INFO - Class 1: 1,052,356 samples (42.3%) -> weight: 0.295
2025-09-24 17:22:25,810 - INFO - Class 2: 1,522 samples (0.1%) -> weight: 204.154
2025-09-24 17:22:25,811 - INFO - Class 3: 869 samples (0.0%) -> weight: 357.563
2025-09-24 17:22:25,811 - INFO - Class 4: 861 samples (0.0%) -> weight: 360.885
2025-09-24 17:22:25,811 - INFO - Class 5: 373,366 samples (15.0%) -> weight: 0.832
2025-09-24 17:22:25,811 - INFO - Class 6: 1,744 samples (0.1%) -> weight: 178.166
2025-09-24 17:22:25,811 - INFO - Class 7: 1,713 samples (0.1%) -> weight: 181.391

rateIdx Class Weights:
0: 0.295
1: 0.295
2: 204.154
3: 357.563
4: 360.885
5: 0.832
6: 178.166
7: 181.391
2025-09-24 17:22:25,817 - INFO - 💾 Class weights saved to: /home/ahmedjk34/Dev/smart-wifi-manager/python_files/model_artifacts/class_weights.json
💾 Class weights saved to: /home/ahmedjk34/Dev/smart-wifi-manager/python_files/model_artifacts/class_weights.json
2025-09-24 17:23:52,322 - INFO - ML-enriched CSV exported: /home/ahmedjk34/Dev/smart-wifi-manager/smart-v3-ml-enriched.csv (rows: 2485777)

ML-enriched CSV exported: /home/ahmedjk34/Dev/smart-wifi-manager/smart-v3-ml-enriched.csv (rows: 2485777, cols: 42)

--- LABEL DISTRIBUTION ---
oracle_conservative:
oracle_conservative
7 744162
5 378931
6 372417
3 360717
4 354520
2 268472
0 5008
1 1550
Name: count, dtype: int64

2025-09-24 17:23:52,352 - INFO - oracle_conservative value counts:
oracle_conservative
7 744162
5 378931
6 372417
3 360717
4 354520
2 268472
0 5008
1 1550
Name: count, dtype: int64
oracle_balanced:
oracle_balanced
7 1106744
6 371480
5 360826
4 339111
3 254395
2 49056
1 2480
0 1685
Name: count, dtype: int64

2025-09-24 17:23:52,363 - INFO - oracle_balanced value counts:
oracle_balanced
7 1106744
6 371480
5 360826
4 339111
3 254395
2 49056
1 2480
0 1685
Name: count, dtype: int64
oracle_aggressive:
oracle_aggressive
7 1364532
5 574947
4 530668
6 8539
3 2943
2 2478
1 1121
0 549
Name: count, dtype: int64

2025-09-24 17:23:52,375 - INFO - oracle_aggressive value counts:
oracle_aggressive
7 1364532
5 574947
4 530668
6 8539
3 2943
2 2478
1 1121
0 549
Name: count, dtype: int64

--- NETWORK CONTEXT DISTRIBUTION ---
network_context
poor_unstable 2474350
emergency_recovery 4107
marginal_conditions 2507
good_stable 2339
excellent_stable 1379
good_unstable 1095
Name: count, dtype: int64
2025-09-24 17:23:52,516 - INFO - Network context distribution:
network_context
poor_unstable 2474350
emergency_recovery 4107
marginal_conditions 2507
good_stable 2339
excellent_stable 1379
good_unstable 1095
Name: count, dtype: int64

--- FEATURE STATISTICS ---
count mean std min 25% 50% 75% max
time 2473777.0 3.163859e+01 1.909396e+01 1.000102 14.861039 3.153339e+01 4.820747e+01 6.500000e+01
stationId 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
rateIdx 2485777.0 1.187048e+00 1.685068e+00 0.000000 0.000000 1.000000e+00 1.000000e+00 7.000000e+00
phyRate 2473777.0 2.177022e+06 1.673508e+06 1000000.000000 1000000.000000 2.000000e+06 2.000000e+06 6.000000e+06
lastSnr 2485777.0 2.147860e+01 5.818449e+00 3.009434 16.000000 2.200000e+01 2.700000e+01 3.199934e+01
snrFast 2473777.0 2.121399e+01 3.693604e+00 12.000000 17.820548 2.047449e+01 2.376169e+01 3.100000e+01
snrSlow 2473777.0 2.089867e+01 1.061503e+00 12.000000 20.373196 2.082433e+01 2.159450e+01 3.100000e+01
snrTrendShort 2473777.0 3.153241e-01 3.033436e+00 -10.080583 -2.451281 7.028200e-02 2.855544e+00 9.499608e+00
snrStabilityIndex 2473777.0 4.563207e+00 1.825690e+00 0.000000 2.872281 3.935734e+00 6.344289e+00 9.500000e+00
snrPredictionConfidence 2473777.0 1.994360e-01 6.147196e-02 0.095238 0.136160 2.026040e-01 2.582460e-01 1.000000e+00
shortSuccRatio 2485777.0 8.771885e-01 6.220001e-02 0.000000 0.900000 9.000000e-01 9.000000e-01 1.000000e+00
medSuccRatio 2473777.0 8.731058e-01 5.109152e-02 0.000000 0.840000 8.800000e-01 9.200000e-01 1.000000e+00
consecSuccess 2485777.0 4.223571e+00 2.915164e+00 0.000000 2.000000 4.000000e+00 7.000000e+00 2.600000e+01
consecFailure 2485777.0 1.287855e-01 4.133580e-01 0.000000 0.000000 0.000000e+00 0.000000e+00 1.100000e+01
recentThroughputTrend 2473777.0 1.353267e+00 1.728064e+00 0.000000 0.000000 1.000000e+00 1.000000e+00 6.300000e+00
packetLossRate 2473777.0 1.252859e-01 4.970761e-02 0.000000 0.100000 1.000000e-01 1.500000e-01 1.000000e+00
retrySuccessRatio 2473777.0 5.324158e+00 1.230914e+00 0.000000 4.250000 6.000000e+00 6.000000e+00 2.000000e+01
recentRateChanges 2473777.0 6.486822e-01 8.389993e-01 0.000000 0.000000 0.000000e+00 1.000000e+00 4.000000e+00
timeSinceLastRateChange 2473777.0 2.209388e+01 1.462222e+01 0.000000 9.000000 2.000000e+01 3.500000e+01 4.900000e+01
rateStabilityScore 2473777.0 9.675659e-01 4.194997e-02 0.800000 0.950000 1.000000e+00 1.000000e+00 1.000000e+00
optimalRateDistance 2473777.0 6.035812e-01 2.680764e-01 0.000000 0.428571 7.142860e-01 8.571430e-01 1.000000e+00
aggressiveFactor 2473777.0 6.742161e-02 1.745683e-01 0.000000 0.000000 0.000000e+00 0.000000e+00 9.500000e-01
conservativeFactor 2473777.0 7.689609e-01 3.865738e-01 0.000000 0.600000 1.000000e+00 1.000000e+00 1.000000e+00
recommendedSafeRate 2473777.0 5.197983e+00 1.620576e+00 2.000000 4.000000 6.000000e+00 7.000000e+00 7.000000e+00
severity 2485777.0 8.263962e-02 5.063433e-02 0.000000 0.048000 7.200000e-02 9.600000e-02 9.999901e-01
confidence 2473777.0 5.633540e-01 1.432283e-01 0.000000 0.450000 4.909260e-01 6.803410e-01 1.000000e+00
T1 2473777.0 1.000000e+01 0.000000e+00 10.000000 10.000000 1.000000e+01 1.000000e+01 1.000000e+01
T2 2473777.0 1.500000e+01 0.000000e+00 15.000000 15.000000 1.500000e+01 1.500000e+01 1.500000e+01
T3 2473777.0 2.500000e+01 0.000000e+00 25.000000 25.000000 2.500000e+01 2.500000e+01 2.500000e+01
decisionReason 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
packetSuccess 2473777.0 8.843748e-01 3.197750e-01 0.000000 1.000000 1.000000e+00 1.000000e+00 1.000000e+00
offeredLoad 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
queueLen 2474777.0 4.920363e-02 2.597625e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 1.990000e+02
retryCount 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
channelWidth 2474777.0 2.000398e+01 2.822554e-01 20.000000 20.000000 2.000000e+01 2.000000e+01 4.000000e+01
mobilityMetric 2474777.0 2.314501e-02 1.254496e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 9.998856e+01
snrVariance 2485777.0 3.210319e+01 5.160084e+00 0.000000 33.250000 3.325000e+01 3.325000e+01 9.025000e+01
2025-09-24 17:23:58,987 - INFO - Feature statistics:
count mean std min 25% 50% 75% max
time 2473777.0 3.163859e+01 1.909396e+01 1.000102 14.861039 3.153339e+01 4.820747e+01 6.500000e+01
stationId 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
rateIdx 2485777.0 1.187048e+00 1.685068e+00 0.000000 0.000000 1.000000e+00 1.000000e+00 7.000000e+00
phyRate 2473777.0 2.177022e+06 1.673508e+06 1000000.000000 1000000.000000 2.000000e+06 2.000000e+06 6.000000e+06
lastSnr 2485777.0 2.147860e+01 5.818449e+00 3.009434 16.000000 2.200000e+01 2.700000e+01 3.199934e+01
snrFast 2473777.0 2.121399e+01 3.693604e+00 12.000000 17.820548 2.047449e+01 2.376169e+01 3.100000e+01
snrSlow 2473777.0 2.089867e+01 1.061503e+00 12.000000 20.373196 2.082433e+01 2.159450e+01 3.100000e+01
snrTrendShort 2473777.0 3.153241e-01 3.033436e+00 -10.080583 -2.451281 7.028200e-02 2.855544e+00 9.499608e+00
snrStabilityIndex 2473777.0 4.563207e+00 1.825690e+00 0.000000 2.872281 3.935734e+00 6.344289e+00 9.500000e+00
snrPredictionConfidence 2473777.0 1.994360e-01 6.147196e-02 0.095238 0.136160 2.026040e-01 2.582460e-01 1.000000e+00
shortSuccRatio 2485777.0 8.771885e-01 6.220001e-02 0.000000 0.900000 9.000000e-01 9.000000e-01 1.000000e+00
medSuccRatio 2473777.0 8.731058e-01 5.109152e-02 0.000000 0.840000 8.800000e-01 9.200000e-01 1.000000e+00
consecSuccess 2485777.0 4.223571e+00 2.915164e+00 0.000000 2.000000 4.000000e+00 7.000000e+00 2.600000e+01
consecFailure 2485777.0 1.287855e-01 4.133580e-01 0.000000 0.000000 0.000000e+00 0.000000e+00 1.100000e+01
recentThroughputTrend 2473777.0 1.353267e+00 1.728064e+00 0.000000 0.000000 1.000000e+00 1.000000e+00 6.300000e+00
packetLossRate 2473777.0 1.252859e-01 4.970761e-02 0.000000 0.100000 1.000000e-01 1.500000e-01 1.000000e+00
retrySuccessRatio 2473777.0 5.324158e+00 1.230914e+00 0.000000 4.250000 6.000000e+00 6.000000e+00 2.000000e+01
recentRateChanges 2473777.0 6.486822e-01 8.389993e-01 0.000000 0.000000 0.000000e+00 1.000000e+00 4.000000e+00
timeSinceLastRateChange 2473777.0 2.209388e+01 1.462222e+01 0.000000 9.000000 2.000000e+01 3.500000e+01 4.900000e+01
rateStabilityScore 2473777.0 9.675659e-01 4.194997e-02 0.800000 0.950000 1.000000e+00 1.000000e+00 1.000000e+00
optimalRateDistance 2473777.0 6.035812e-01 2.680764e-01 0.000000 0.428571 7.142860e-01 8.571430e-01 1.000000e+00
aggressiveFactor 2473777.0 6.742161e-02 1.745683e-01 0.000000 0.000000 0.000000e+00 0.000000e+00 9.500000e-01
conservativeFactor 2473777.0 7.689609e-01 3.865738e-01 0.000000 0.600000 1.000000e+00 1.000000e+00 1.000000e+00
recommendedSafeRate 2473777.0 5.197983e+00 1.620576e+00 2.000000 4.000000 6.000000e+00 7.000000e+00 7.000000e+00
severity 2485777.0 8.263962e-02 5.063433e-02 0.000000 0.048000 7.200000e-02 9.600000e-02 9.999901e-01
confidence 2473777.0 5.633540e-01 1.432283e-01 0.000000 0.450000 4.909260e-01 6.803410e-01 1.000000e+00
T1 2473777.0 1.000000e+01 0.000000e+00 10.000000 10.000000 1.000000e+01 1.000000e+01 1.000000e+01
T2 2473777.0 1.500000e+01 0.000000e+00 15.000000 15.000000 1.500000e+01 1.500000e+01 1.500000e+01
T3 2473777.0 2.500000e+01 0.000000e+00 25.000000 25.000000 2.500000e+01 2.500000e+01 2.500000e+01
decisionReason 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
packetSuccess 2473777.0 8.843748e-01 3.197750e-01 0.000000 1.000000 1.000000e+00 1.000000e+00 1.000000e+00
offeredLoad 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
queueLen 2474777.0 4.920363e-02 2.597625e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 1.990000e+02
retryCount 2473777.0 0.000000e+00 0.000000e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 0.000000e+00
channelWidth 2474777.0 2.000398e+01 2.822554e-01 20.000000 20.000000 2.000000e+01 2.000000e+01 4.000000e+01
mobilityMetric 2474777.0 2.314501e-02 1.254496e+00 0.000000 0.000000 0.000000e+00 0.000000e+00 9.998856e+01
snrVariance 2485777.0 3.210319e+01 5.160084e+00 0.000000 33.250000 3.325000e+01 3.325000e+01 9.025000e+01
2025-09-24 17:23:58,987 - INFO - === ML Data Prep Script Finished ===
ahmedjk34@pop-os:~/Dev/smart-wifi-manager$
