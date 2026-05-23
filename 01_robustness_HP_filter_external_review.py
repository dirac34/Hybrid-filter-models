"""
================================================================
ROBUSTNESS ANALYSIS - HP FILTER (Computational Economics paper)
================================================================
Versión corregida que aborda las críticas metodológicas:

  1. División cronológica con walk-forward expanding window
     (sustituye al train_test_split aleatorio del paper original)
  2. HP filter aplicado de forma CAUSAL en cada punto de test
     (sustituye al HP aplicado sobre toda la serie ex-ante)
  3. Scaler ajustado únicamente sobre datos de training en cada fold
     (sustituye al scaler ajustado sobre toda la serie)

Arquitectura de modelos: idéntica al paper publicado para
garantizar comparabilidad directa de resultados.

Para análisis de robustez post-publicación.
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
from statsmodels.tsa.filters.hp_filter import hpfilter
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

DATA_PATH = r"C:\Users\elect\OneDrive\Documentos\Doctorado\Articulo 2\Data\Inflation_Parameters.csv"
OUTPUT_PATH = r"robustness_results_HP_walkforward.csv"

# Mantengo la estructura del paper HP: inflación entra como predictor candidato
# (esto es diferente de los otros scripts BK/Kalman donde solo se usa como target)
ALL_FEATURES = ['USM2', 'USPPIYY', 'INDPRO', 'UNRATE', 'USOIL', 'Annual Inflation Rate']
TARGET = 'Annual Inflation Rate'
PREDICTORS = [f for f in ALL_FEATURES if f != TARGET]

# Parámetros (idénticos al paper)
SEQ_LENGTH = 10
HP_LAMBDA = 129600        # Lambda estándar para datos mensuales
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

# Walk-forward
INITIAL_TRAIN_FRACTION = 0.60
TEST_FOLD_SIZE = 12

# ============================================================
# FILTRO CAUSAL
# ============================================================

def causal_hp_filter(full_series, train_end_idx, test_end_idx, lamb=HP_LAMBDA):
    """
    HP filter aplicado respetando la causalidad temporal.

    - Tramo training [0, train_end_idx]: HP estándar (no hay leakage ya que
      toda esta información está disponible en el momento de entrenar).
    - Cada punto t del test: HP aplicado a [0, t+1] conservando solo el último
      valor como estimación causal de la tendencia en t.
    """
    n = len(full_series)
    causal_trend = np.zeros(n)

    _, train_trend = hpfilter(full_series[:train_end_idx], lamb=lamb)
    causal_trend[:train_end_idx] = train_trend

    for t in range(train_end_idx, test_end_idx):
        _, trend_t = hpfilter(full_series[:t + 1], lamb=lamb)
        last_val = trend_t.iloc[-1] if hasattr(trend_t, 'iloc') else trend_t[-1]
        causal_trend[t] = last_val

    return causal_trend

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def create_sequences(data, seq_length):
    """
    Crea secuencias para predicción. Mantiene la estructura del paper HP:
    y[i] = data[i + seq_length - 1, -1] (target contemporáneo al final de la secuencia).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])
        y.append(data[i + seq_length - 1, -1])
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
# CONSTRUCTORES DE MODELOS (idénticos al paper publicado)
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

        # HP causal sobre cada columna
        filtered_data = data_sub.copy()
        for col in selected_cols:
            filtered_data[col] = causal_hp_filter(
                data_sub[col].values, current_train_end, test_end, lamb=HP_LAMBDA
            )

        train_raw = filtered_data.iloc[:current_train_end].values
        full_raw = filtered_data.iloc[:test_end].values

        # Scaler ajustado SOLO con training
        scaler_fold = scaler_class()
        scaler_fold.fit(train_raw)
        train_scaled = scaler_fold.transform(train_raw)
        full_scaled = scaler_fold.transform(full_raw)

        X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
        X_all, y_all = create_sequences(full_scaled, SEQ_LENGTH)

        test_start_idx = current_train_end - SEQ_LENGTH + 1
        test_end_idx = test_end - SEQ_LENGTH + 1
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
    print(" ROBUSTNESS ANALYSIS - HP FILTER")
    print(" Walk-forward expanding window | Causal HP | Train-only scaler")
    print("=" * 70)

    merged_data = pd.read_csv(DATA_PATH)
    if merged_data.isnull().sum().sum() > 0:
        merged_data = merged_data.interpolate(method='linear')
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    print(f"\nDataset: {merged_data.shape[0]} meses, {merged_data.shape[1]} variables")

    scalers = {'MinMax': MinMaxScaler, 'Standard': StandardScaler}
    all_results = []

    total_combos = sum(1 for r in range(1, len(PREDICTORS) + 1)
                       for _ in combinations(PREDICTORS, r))
    total_exp = total_combos * len(scalers)
    exp_idx = 0
    t_start = time.time()

    for r in range(1, len(PREDICTORS) + 1):
        for combo in combinations(PREDICTORS, r):
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
                        'filter': 'HP_causal',
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
