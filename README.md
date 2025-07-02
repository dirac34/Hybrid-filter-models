# Hybrid-filter-models

# Inflation Forecasting with Time-Series Filtering and Neural Networks

This repository contains the implementation code for the research paper "Enhancement of Inflation Forecasting Accuracy through Advanced Time-Series Filtering and Machine Learning Integration". The study examines the performance of various filtering techniques combined with neural network architectures for inflation prediction using U.S. Federal Reserve economic indicators.

## 📁 Repository Structure

```
Code/
├── neural_network_model_baseline_review.ipynb       # Baseline models (no filtering)
├── neural_network_model_bk_filter_review.ipynb     # Baxter-King filtering models
├── neural_network_model_hp_filter_review_2.ipynb   # Hodrick-Prescott filtering models
└── neural_network_model_kf_filter_review_2.ipynb   # Kalman filtering models
```

## 📊 Notebooks Overview

### 1. `neural_network_model_baseline_review.ipynb`
**Baseline Models - No Filtering**
- Implements neural network models on raw Federal Reserve data
- Serves as the performance baseline for comparison
- Includes ANN, RNN, LSTM, GRU, CNN architectures
- Demonstrates the challenges of working with unprocessed economic data

### 2. `neural_network_model_bk_filter_review.ipynb`
**Baxter-King Band-Pass Filtering**
- Applies Baxter-King filtering to isolate business cycle frequencies (6-32 quarters)
- Tests neural network performance on band-pass filtered data
- Explores frequency domain preprocessing for inflation forecasting
- Evaluates the effectiveness of business cycle isolation

### 3. `neural_network_model_hp_filter_review_2.ipynb`
**Hodrick-Prescott Filtering**
- Implements HP filtering with λ = 129,600 for monthly data
- Demonstrates superior performance with trend-cycle decomposition
- Achieves near-perfect forecasting accuracy (R² > 0.99)
- Shows transformational improvement in neural network performance

### 4. `neural_network_model_kf_filter_review_2.ipynb`
**Multivariate Kalman Filtering**
- Applies state-space models with EM parameter estimation
- Tests multivariate cross-correlation modeling
- Explores advanced statistical filtering techniques
- Examines stability and performance trade-offs

## 🔧 Key Features

- **Multiple Architectures**: ANN, RNN, LSTM, GRU, CNN implementations
- **Comprehensive Filtering**: HP, Baxter-King, and Kalman filtering methods
- **Performance Metrics**: MSE, MAE, R², MAPE evaluation
- **Scaling Methods**: MinMax and Standard scaling comparisons
- **Feature Selection**: Analysis of 2-3 key economic indicators
- **Time Series Analysis**: Monthly U.S. data from 1959-2024

## 📈 Key Findings

- **HP Filtering Superiority**: 60-100× performance improvements over alternatives
- **Feature Parsimony**: 2-3 indicators outperform high-dimensional approaches
- **Producer Price Index**: Single best predictor (84% variance explained)
- **Preprocessing Dominance**: Method quality matters more than model complexity
- **Neural Network Transformation**: From failure to competitive performance

## 🚀 Getting Started

### Prerequisites
```python
pandas
numpy
scikit-learn
tensorflow/keras
matplotlib
seaborn
statsmodels
scipy
```

### Usage
1. Clone the repository
2. Install required dependencies
3. Run notebooks in the following order:
   - Start with `baseline_review.ipynb` to establish benchmarks
   - Compare with filtering approaches: `hp_filter`, `bk_filter`, `kf_filter`
   - Analyze performance differences across methods

## 📊 Data Requirements

- Monthly U.S. Federal Reserve economic indicators (FRED database)
- Target variable: Consumer Price Index inflation (CPIAUCSL)
- Key predictors: USPPIYY, USM2, INDPRO, UNRATE
- Time period: 1959-2024

## 🔍 Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## 📝 Research Applications

This code supports research in:
- Macroeconomic forecasting
- Time-series preprocessing optimization
- Neural network applications in economics
- Monetary policy analysis
- Central banking applications

## 🎯 Performance Highlights

| Method | Best R² | Best MSE | Architecture |
|--------|---------|----------|--------------|
| HP Filter | >0.999 | <0.001 | Ensemble |
| Baseline | 0.84 | ~0.16 | Single Indicator |
| BK Filter | 0.92 | ~0.08 | Limited Success |
| Kalman | Unstable | High | Parameter Sensitive |

## 📚 Citation

If you use this code in your research, please cite:
```
[Paper Title: Analysis of improvements in inflation prediction performance
through the use of hybrid filter models and many-to-one
neural networks]
[Authors, Journal, Year]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Note**: This research demonstrates that sophisticated preprocessing through HP filtering can achieve transformational improvements in inflation forecasting, with implications for monetary policy and economic planning.
