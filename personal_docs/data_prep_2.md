# Smart WiFi Manager ML Training Data Generation

A comprehensive NS-3 simulation framework for generating high-quality machine learning training datasets focused on WiFi rate adaptation and network performance analysis. This project generates diverse networking scenarios with emphasis on challenging conditions to train robust Random Forest, XGBoost, and other ML models.

## 🎯 Project Overview

This simulation framework was developed to address the critical need for comprehensive training data in WiFi rate adaptation machine learning models. Traditional datasets often lack sufficient coverage of poor and medium performance scenarios, leading to models that perform well only under ideal conditions.

### Key Features

- **400 Diverse Test Scenarios** - Strategically distributed across performance categories
- **75% Focus on Challenging Cases** - Emphasizes poor and medium performance scenarios
- **21+ Features per Scenario** - Rich feature set for comprehensive ML training
- **800K-1.2M Decision Points** - Extensive dataset for robust model training
- **Disk-Efficient Logging** - Minimal, stratified, and progress-aware logs
- **Real-time Progress Tracking** - 10% increment progress updates

## 📊 Dataset Composition

### Scenario Distribution (400 Total Cases)

- **120 Poor Performance** (30%) - SNR 3-12 dB, high mobility, multiple interferers
- **160 Medium Performance** (40%) - SNR 12-22 dB, moderate conditions
- **60 High Interference** (15%) - 4-7 interferers, challenging RF environment
- **60 Good Performance** (15%) - SNR 22-30 dB, baseline scenarios

### Parameter Coverage

- **SNR Range**: 3-30 dB (75% emphasis on challenging 3-22 dB range)
- **Mobility**: 0.5-20 m/s (higher speeds for poor performance scenarios)
- **Interferers**: 1-7 concurrent interfering networks
- **Packet Sizes**: 256-3072 bytes (larger packets for challenging scenarios)
- **Traffic Rates**: 20-55 Mbps (adaptive based on scenario difficulty)

## 🏗️ Architecture

### Core Components

#### 1. Performance-Based Parameter Generator (`performance-based-parameter-generator.h/cpp`)
- Stratifies and generates scenarios across four performance categories.
- Easily customizable for research and ablation studies.

#### 2. Decision Count Controller (`decision-count-controller.h/cpp`)
- Adaptive, periodic, and emergency stop logic to maximize decision data and minimize wasted compute.
- Logs simulation summaries as comments, not dummy data rows.

#### 3. MinstrelWifiManagerLogged (`minstrel-wifi-manager-logged.cc`)
- Instrumented Minstrel variant with feedback-oriented, stratified probabilistic logging.
- Traces rate-change and per-packet adaptation events for ML.
- Logs >20 features per decision for direct ML consumption.

#### 4. Main Simulation Engine (`main-simulation-fixed.cpp`)
- Orchestrates the scenario generation, data collection, and progress reporting.

## 🚀 Getting Started

See the original message above for full prerequisites, installation, build, and quick start instructions.

## 🛠️ Development Journey & Problem Solving

**(See original message above for full details, issue breakdowns, and troubleshooting.)**

## 📊 Output Data Format

See main message for the full CSV column schema.

## 🤖 ML Model Compatibility

Ideal for tabular ML models (Random Forest, XGBoost, etc.), but also compatible with neural networks given adequate scaling.

## 🔧 Customization Options

Adjust scenario counts, parameter ranges, and category splits directly in the scenario generator for research.

## 🐛 Troubleshooting

See full README for simulation, disk, and build troubleshooting.

## 📚 Research Applications

WiFi rate adaptation, network performance prediction, resource allocation, anomaly detection, and more.

## 🤝 Contributing

See main message for detailed contribution guidelines and roadmap.

## 📄 License

This project is licensed under the **MIT License**.

---
