"""
================================================================
ROBUSTNESS ANALYSIS - BAXTER-KING FILTER
================================================================
Versión corregida que aborda las críticas metodológicas:

  1. División cronológica con walk-forward expanding window
     (sustituye al train_test_split aleatorio)
  2. BK filter aplicado de forma CAUSAL en cada punto de test
     PROBLEMA del BK estándar: por diseño usa K observaciones FUTURAS
     SOLUCIÓN: BK simétrico solo en training; BK asimétrico (solo lags
     pasados) en cada punto de test → completamente causal
  3. Scaler ajustado únicamente sobre datos de training en cada fold

NOTA TÉCNICA SOBRE EL BK ASIMÉTRICO:
El BK estándar usa pesos simétricos w_{-K}, ..., w_0, ..., w_K que
multiplican observaciones t-K, ..., t, ..., t+K. Esto es inherentemente
no-causal. Para hacerlo causal en test, usamos solo w_0, w_{-1}, ..., w_{-K}
aplicados a t, t-1, ..., t-K. Re-normalizamos los pesos para que sumen 0
(condición necesaria para que sea un filtro de business cycle válido).
================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dense, Conv1D, Flatten, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from itertools import combinations
import warnings
import time
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_PATH = r"C:\Users\elect\OneDrive\Documentos\Doctorado\Articulo 2 peer review\Data\Inflation_Parameters.csv"
OUTPUT_PATH = r"robustness_results_BK_walkforward.csv"

# Estructura del paper BK: no incluye inflación como predictor
LEADING_INDICATORS = ['USM2', 'USPPIYY', 'INDPRO', 'UNRATE', 'USOIL']
TARGET = 'Annual Inflation Rate'

# Parámetros (idénticos al paper BK)
SEQ_LENGTH = 50
BK_LOW_FREQ = 6
BK_HIGH_FREQ = 32
BK_K = 12
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

INITIAL_TRAIN_FRACTION = 0.60
TEST_FOLD_SIZE = 12

# ============================================================
# BK CAUSAL FILTER
# ============================================================

def compute_bk_weights(low_freq, high_freq, K):
    """
    Calcula los pesos del filtro Baxter-King estándar (simétrico).
    Devuelve un array de 2K+1 pesos.
    """
    omega_low = 2 * np.pi / high_freq
    omega_high = 2 * np.pi / low_freq

    weights = np.zeros(2 * K + 1)
    weights[K] = (omega_high - omega_low) / np.pi

    for k in range(1, K + 1):
        wk = (np.sin(k * omega_high) - np.sin(k * omega_low)) / (k * np.pi)
        weights[K + k] = wk
        weights[K - k] = wk

    # Normalizar para que sumen cero (filtro de business cycle válido)
    weight_sum = np.sum(weights)
    if abs(weight_sum) > 1e-10:
        weights[K] -= weight_sum

    return weights


def apply_bk_symmetric(data, low_freq=BK_LOW_FREQ, high_freq=BK_HIGH_FREQ, K=BK_K):
    """
    BK simétrico estándar. Se usa SOLO sobre el periodo de training,
    donde no hay riesgo de leakage hacia test.
    """
    weights = compute_bk_weights(low_freq, high_freq, K)
    padded = np.pad(data, K, mode='edge')
    filtered = np.convolve(padded, weights, mode='valid')

    if len(filtered) != len(data):
        if len(filtered) > len(data):
            start = (len(filtered) - len(data)) // 2
            filtered = filtered[start:start + len(data)]
        else:
            pad = len(data) - len(filtered)
            filtered = np.pad(filtered, (pad // 2, pad - pad // 2), mode='edge')
    return filtered


def apply_bk_causal_single_point(series_up_to_t, low_freq=BK_LOW_FREQ,
                                  high_freq=BK_HIGH_FREQ, K=BK_K):
    """
    BK asimétrico (solo lags pasados) para un único punto t.

    Toma la serie hasta t (inclusive) y aplica los pesos del BK pero
    SOLO los pesos de lags pasados w_0, w_{-1}, ..., w_{-K} sobre las
    últimas K+1 observaciones. Re-normaliza los pesos para que sumen 0.

    Devuelve la estimación causal del componente cíclico en t.
    """
    full_weights = compute_bk_weights(low_freq, high_freq, K)
    causal_weights = full_weights[:K + 1].copy()  # w_{-K} ... w_0

    # Re-normalizar para que sumen cero
    weight_sum = np.sum(causal_weights)
    if abs(weight_sum) > 1e-10:
        causal_weights[-1] -= weight_sum  # ajusta w_0

    n = len(series_up_to_t)
    if n < K + 1:
        # Si no hay suficientes lags, usar lo que haya
        n_avail = min(n, K + 1)
        w_avail = causal_weights[-n_avail:]
        w_avail = w_avail - np.mean(w_avail)  # re-centrar
        return np.sum(series_up_to_t[-n_avail:] * w_avail)

    # Aplicar los pesos a las últimas K+1 observaciones
    return np.sum(series_up_to_t[-(K + 1):] * causal_weights)


def causal_bk_filter(full_series, train_end_idx, test_end_idx,
                     low_freq=BK_LOW_FREQ, high_freq=BK_HIGH_FREQ, K=BK_K):
    """
    BK aplicado respetando causalidad temporal.
    - Training [0, train_end_idx]: BK simétrico estándar
    - Cada t en test: BK asimétrico usando solo datos [0, t]
    """
    n = len(full_series)
    causal_cycle = np.zeros(n)

    # Training: BK simétrico estándar
    causal_cycle[:train_end_idx] = apply_bk_symmetric(
        full_series[:train_end_idx], low_freq, high_freq, K
    )

    # Test: BK asimétrico punto a punto
    for t in range(train_end_idx, test_end_idx):
        causal_cycle[t] = apply_bk_causal_single_point(
            full_series[:t + 1], low_freq, high_freq, K
        )

    return causal_cycle

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def create_sequences(data, seq_length):
    """
    Estructura del paper BK: y[i] = data[i + seq_length, -1]
    (predicción a t+1, no contemporánea).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])
        y.append(data[i + seq_length, -1])
    return np.array(X), np.array(y)


def calculate_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

# ============================================================
# MODELOS (idénticos al paper BK)
# ============================================================

def build_lstm(input_shape):
    m = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m

def build_gru(input_shape):
    m = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.2),
        GRU(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m

def build_cnn(input_shape):
    m = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m

def build_rnn(input_shape):
    m = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.2),
        SimpleRNN(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m

def build_ann(input_shape):
    m = Sequential([
        Dense(50, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m

# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def walk_forward_evaluate(merged_data, feature_combo, scaler_class):
    selected_cols = list(feature_combo) + [TARGET]
    data_sub = merged_data[selected_cols].copy()
    n = len(data_sub)
    initial_train_end = int(n * INITIAL_TRAIN_FRACTION)

    model_names = ['LSTM', 'GRU', 'CNN', 'RNN', 'ANN', 'RandomForest', 'XGBoost']
    fold_metrics = {m: [] for m in model_names}

    current_train_end = initial_train_end
    fold_idx = 0

    while current_train_end + TEST_FOLD_SIZE <= n:
        fold_idx += 1
        test_end = current_train_end + TEST_FOLD_SIZE

        # BK causal sobre cada columna
        filtered_data = data_sub.copy()
        for col in selected_cols:
            filtered_data[col] = causal_bk_filter(
                data_sub[col].values, current_train_end, test_end
            )

        train_raw = filtered_data.iloc[:current_train_end].values
        full_raw = filtered_data.iloc[:test_end].values

        scaler_fold = scaler_class()
        scaler_fold.fit(train_raw)
        train_scaled = scaler_fold.transform(train_raw)
        full_scaled = scaler_fold.transform(full_raw)

        X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
        X_all, y_all = create_sequences(full_scaled, SEQ_LENGTH)

        # Para BK seq: y[i] = data[i+seq_length, -1] → target en posición i+seq_length
        test_start_idx = current_train_end - SEQ_LENGTH
        test_end_idx = test_end - SEQ_LENGTH
        test_start_idx = max(test_start_idx, 0)
        test_end_idx = min(test_end_idx, len(X_all))
        X_test = X_all[test_start_idx:test_end_idx]
        y_test = y_all[test_start_idx:test_end_idx]

        if len(X_test) == 0 or len(X_train) < 20:
            current_train_end += TEST_FOLD_SIZE
            continue

        input_shape = (SEQ_LENGTH, X_train.shape[2])
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        try:
            builders = {
                'LSTM': (build_lstm, X_train, X_test, input_shape),
                'GRU': (build_gru, X_train, X_test, input_shape),
                'CNN': (build_cnn, X_train, X_test, input_shape),
                'RNN': (build_rnn, X_train, X_test, input_shape),
                'ANN': (build_ann, X_train_flat, X_test_flat, (X_train_flat.shape[1],))
            }
            for mname, (builder, Xtr, Xte, shape) in builders.items():
                m = builder(shape)
                es = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                   restore_best_weights=True)
                m.fit(Xtr, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.2, verbose=0, callbacks=[es])
                pred = m.predict(Xte, verbose=0).flatten()
                fold_metrics[mname].append(calculate_metrics(y_test, pred))

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_flat, y_train)
            fold_metrics['RandomForest'].append(
                calculate_metrics(y_test, rf.predict(X_test_flat))
            )

            xgb_m = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_m.fit(X_train_flat, y_train)
            fold_metrics['XGBoost'].append(
                calculate_metrics(y_test, xgb_m.predict(X_test_flat))
            )
        except Exception as e:
            print(f"      ⚠ Error fold {fold_idx}: {e}")

        current_train_end += TEST_FOLD_SIZE

    aggregated = {}
    for model, flist in fold_metrics.items():
        if flist:
            aggregated[model] = {
                'mse_mean': np.mean([f['mse'] for f in flist]),
                'mse_std': np.std([f['mse'] for f in flist]),
                'rmse_mean': np.mean([f['rmse'] for f in flist]),
                'mae_mean': np.mean([f['mae'] for f in flist]),
                'r2_mean': np.mean([f['r2'] for f in flist]),
                'r2_std': np.std([f['r2'] for f in flist]),
                'mape_mean': np.mean([f['mape'] for f in flist]),
                'n_folds': len(flist)
            }
    return aggregated, fold_idx

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print(" ROBUSTNESS ANALYSIS - BAXTER-KING FILTER")
    print(" Walk-forward expanding window | BK causal asimétrico | Train-only scaler")
    print("=" * 70)

    merged_data = pd.read_csv(DATA_PATH)
    if merged_data.isnull().sum().sum() > 0:
        merged_data = merged_data.interpolate(method='linear')
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    print(f"\nDataset: {merged_data.shape[0]} meses, {merged_data.shape[1]} variables")
    print(f"BK config: cycle range {BK_LOW_FREQ}-{BK_HIGH_FREQ}Q, K={BK_K}")

    scalers = {'MinMax': MinMaxScaler, 'Standard': StandardScaler}
    all_results = []

    total_combos = sum(1 for r in range(1, len(LEADING_INDICATORS) + 1)
                       for _ in combinations(LEADING_INDICATORS, r))
    total_exp = total_combos * len(scalers)
    exp_idx = 0
    t_start = time.time()

    for r in range(1, len(LEADING_INDICATORS) + 1):
        for combo in combinations(LEADING_INDICATORS, r):
            for scaler_name, scaler_class in scalers.items():
                exp_idx += 1
                elapsed = (time.time() - t_start) / 60
                print(f"\n[{exp_idx}/{total_exp}] ({elapsed:.1f}min) "
                      f"{combo} | {scaler_name}")

                results, n_folds = walk_forward_evaluate(merged_data, combo, scaler_class)
                print(f"   Folds: {n_folds}")

                for model_name, metrics in results.items():
                    row = {
                        'features': '+'.join(combo),
                        'n_features': len(combo),
                        'scaler': scaler_name,
                        'model': model_name,
                        'filter': 'BK_causal_asymmetric',
                        **metrics
                    }
                    all_results.append(row)
                    print(f"   {model_name:14s} R²={metrics['r2_mean']:.4f}±{metrics['r2_std']:.4f}  "
                          f"MSE={metrics['mse_mean']:.6f}")

                pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False)

    results_df = pd.DataFrame(all_results)
    print(f"\n✓ Guardado en: {OUTPUT_PATH}")
    print("\nRanking de modelos (R² medio):")
    print(results_df.groupby('model')[['r2_mean', 'mse_mean']].mean()
          .sort_values('r2_mean', ascending=False).round(4))
    print(f"\nTiempo total: {(time.time() - t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
