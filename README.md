# Hybrid-filter-models

# Advanced Economic Forecasting with Time Series Filtering

## Overview

This repository provides a comprehensive implementation of advanced time series filtering techniques for economic forecasting, specifically designed for inflation prediction using Federal Reserve indicators. The system combines multiple filtering methods with machine learning and deep learning models to deliver state-of-the-art forecasting performance.

## Key Features

### Core Capabilities
- **Hodrick-Prescott (HP) Filter** - Trend-cycle decomposition for economic time series
- **Baxter-King (BK) Filter** - Band-pass filtering for business cycle extraction
- **Kalman Filter** - Optimal state estimation with uncertainty quantification
- **Multi-Model Ensemble** - LSTM, GRU, CNN, RNN, ANN, Random Forest, XGBoost
- **Comprehensive Metrics** - MSE, RMSE, MAE, R², MAPE evaluation
- **Advanced Preprocessing** - MinMax/Standard scaling with missing value handling
- **Publication-Ready Visualizations** - Professional plots for research papers

### Supported Economic Indicators
```python
ECONOMIC_INDICATORS = {
    'USM2': 'M2 Money Supply',
    'USPPIYY': 'Producer Price Index (Year-over-Year)',
    'INDPRO': 'Industrial Production Index', 
    'UNRATE': 'Unemployment Rate',
    'USOIL': 'Crude Oil Prices WTI'
}
TARGET = 'Annual Inflation Rate'
```

## Architecture

### Model Configurations
```python
MODEL_CONFIGS = {
    'LSTM': {
        'units': 50,
        'activation': 'tanh',
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'layers': 2,
        'architecture': 'Sequential with 2 LSTM layers + Dense output'
    },
    'GRU': {
        'units': 50,
        'activation': 'tanh', 
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'layers': 2,
        'architecture': 'Sequential with 2 GRU layers + Dense output'
    },
    'CNN': {
        'filters': 64,
        'kernel_size': 2,
        'activation': 'relu',
        'dense_units': 50,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'architecture': 'Conv1D + Flatten + Dense layers'
    },
    'Random_Forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'architecture': 'Ensemble of decision trees'
    },
    'XGBoost': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'architecture': 'Gradient boosting framework'
    }
}
```

## Installation

```bash
# Clone repository
git clone https://github.com/dirac34/Hybrid-filter-models.git


# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
xgboost>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
statsmodels>=0.13.0
scipy>=1.7.0
```

## Usage

### Quick Start
```python
import pandas as pd
from economic_forecasting import EconomicForecaster

# Initialize forecaster with HP filter
forecaster = EconomicForecaster(
    filter_type='hp',
    lambda_param=129600,  # Monthly data
    sequence_length=50
)

# Load data
data = pd.read_csv('inflation_parameters.csv')

# Run comprehensive analysis
results = forecaster.run_analysis(
    data=data,
    target_column='Annual Inflation Rate',
    test_size=0.2,
    scalers=['MinMax', 'Standard'],
    models=['LSTM', 'GRU', 'CNN', 'Random_Forest', 'XGBoost']
)

# Generate report
forecaster.generate_report(results, save_path='forecasting_results.html')
```

### Data Preprocessing Pipeline

```python
def load_and_document_data(file_path):
    """
    Comprehensive data preprocessing with documentation
    
    Steps:
    1. Load CSV data
    2. Handle missing values via interpolation
    3. Apply HP filter for trend extraction
    4. Normalize using MinMax/Standard scaling
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.interpolate(method='linear')
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Apply HP filter
    for column in data.columns:
        if column != 'date':
            filtered_data = apply_hp_filter(data[column].values)
            data[column] = filtered_data
    
    return data
```

### Model Training Example

```python
# Build LSTM model with dropout regularization
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Train with early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)
```

## Filtering Techniques

### 1. Hodrick-Prescott Filter

```python
def apply_hp_filter(data, lamb=129600):
    """
    Apply HP filter for trend extraction
    
    Parameters:
    -----------
    data : array-like
        Time series data
    lamb : float
        Smoothing parameter (129600 for monthly data)
    
    Returns:
    --------
    np.array : Trend component
    """
    try:
        cycle, trend = hpfilter(data, lamb=lamb)
        return trend
    except Exception as e:
        print(f"HP Filter warning: {e}")
        return data
```

### 2. Baxter-King Filter

```python
def apply_bk_filter(data, low_freq=6, high_freq=32, K=12):
    """
    Apply Baxter-King band-pass filter
    
    Parameters:
    -----------
    low_freq : int
        Minimum cycle length (quarters)
    high_freq : int  
        Maximum cycle length (quarters)
    K : int
        Filter length
    """
    # Implementation for business cycle extraction
    pass
```

### 3. Kalman Filter

```python
def apply_kalman_filter(data, process_noise=0.01, observation_noise=0.1):
    """
    Apply Kalman filter for optimal state estimation
    
    State equation: x_t = F_t * x_{t-1} + w_t
    Observation:    z_t = H_t * x_t + v_t
    """
    # Implementation for real-time filtering
    pass
```

## Performance Metrics

### Comprehensive Evaluation
```python
def calculate_comprehensive_metrics(y_true, y_pred):
    """
    Calculate multiple evaluation metrics
    
    Returns:
    --------
    dict : All performance metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
    }
```

## Results Analysis

### Visualization Examples

```python
# Performance comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# MSE by model type
model_mse = results_df.groupby('model')['mse'].mean().sort_values()
axes[0,0].bar(range(len(model_mse)), model_mse.values)
axes[0,0].set_title('MSE by Model Type')

# R² by normalization method
scaler_perf = results_df.groupby(['model', 'scaler'])['r2'].mean().unstack()
scaler_perf.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Normalization Impact')

# Feature count vs performance
feature_impact = results_df.groupby('n_features')['r2'].agg(['mean', 'std'])
axes[0,2].errorbar(feature_impact.index, feature_impact['mean'], 
                   yerr=feature_impact['std'])
axes[0,2].set_title('Performance vs Feature Count')

plt.tight_layout()
plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
```

## Research Applications

### Central Banking
- **Inflation forecasting** for monetary policy
- **Economic indicator analysis** 
- **Real-time economic monitoring**
- **Risk assessment** and scenario planning

### Financial Markets
- **Asset price prediction**
- **Portfolio optimization**
- **Risk management** systems
- **Market volatility** forecasting

### Policy Analysis
- **Fiscal policy** impact assessment
- **Economic growth** projections
- **Business cycle** identification
- **Macroeconomic** modeling

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Code formatting
black src/
flake8 src/

# Type checking
mypy src/
```

