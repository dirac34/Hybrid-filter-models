"""
================================================================
ROBUSTNESS ANALYSIS - KALMAN FILTER (Multivariate)
================================================================
Versión corregida que aborda las críticas metodológicas:

  1. División cronológica con walk-forward expanding window
     (sustituye al K-fold mal etiquetado como "walk-forward")

  2. Kalman FILTER en lugar de Kalman SMOOTHER
     PROBLEMA del smoother: kf.smooth() estima P(x_t | y_1,...,y_T)
     usando información de TODA la serie, incluyendo futuro → leakage.
     SOLUCIÓN: kf.filter() estima P(x_t | y_1,...,y_t) → causal por
     construcción.

  3. EM solo sobre datos de training en cada fold
     PROBLEMA: el EM original ajusta parámetros del Kalman usando toda
     la serie, lo que contamina con información futura.
     SOLUCIÓN: kf.em() entrenado solo sobre el training window, y luego
     kf.filter() aplicado expansivamente sobre el periodo de test.

  4. Scaler ajustado únicamente sobre datos de training en cada fold

NOTA: este script omite las funciones de "sequence length sensitivity"
y "EM parameter sensitivity" del original porque introducen leakage si se
ejecutan sobre toda la serie. Se mantienen los parámetros del paper.
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
from pykalman import KalmanFilter
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
OUTPUT_PATH = r"robustness_results_Kalman_walkforward.csv"

# Estructura del paper Kalman: no incluye inflación como predictor
LEADING_INDICATORS = ['USM2', 'USPPIYY', 'INDPRO', 'UNRATE', 'USOIL']
TARGET = 'Annual Inflation Rate'

# Parámetros (idénticos al paper)
SEQ_LENGTH = 50
N_ITER_EM = 10
BURN_IN = 10
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

INITIAL_TRAIN_FRACTION = 0.60
TEST_FOLD_SIZE = 12

# ============================================================
# KALMAN CAUSAL FILTER
# ============================================================

def causal_kalman_filter(data, train_end_idx, test_end_idx,
                         n_iter_em=N_ITER_EM, burn_in=BURN_IN):
    """
    Kalman multivariante aplicado respetando la causalidad temporal.

    1. Estimar parámetros con EM SOLO sobre el periodo de training.
       Esto evita que las matrices de covarianza incorporen información
       de la dinámica de los datos de test.

    2. Aplicar kf.filter() (no smoother) a toda la ventana [0, test_end_idx]
       usando los parámetros estimados. El filter es causal: la estimación
       del estado en t solo usa observaciones [0, t].

    Returns:
        filtered_states : array (test_end_idx - burn_in, n_dim)
    """
    n_dim = data.shape[1]
    obs_train = data[:train_end_idx].values
    obs_full = data[:test_end_idx].values

    # Inicializar Kalman con configuración multivariante
    kf = KalmanFilter(
        n_dim_obs=n_dim,
        n_dim_state=n_dim,
        transition_matrices=np.eye(n_dim),
        observation_matrices=np.eye(n_dim),
        initial_state_mean=np.zeros(n_dim),
        initial_state_covariance=np.eye(n_dim) * 1e3
    )

    # EM SOLO con training data (sin leakage)
    kf = kf.em(obs_train, n_iter=n_iter_em)

    # FILTER (causal) sobre toda la ventana usando parámetros entrenados
    # filter_update aplica el paso del filtro de Kalman secuencialmente
    state_means, state_covs = kf.filter(obs_full)

    # Discard burn-in
    if burn_in > 0 and burn_in < state_means.shape[0]:
        state_means = state_means[burn_in:, :]

    return state_means

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def create_sequences(data, seq_length):
    """
    Estructura del paper Kalman: y[i] = data[i + seq_length, -1]
    (predicción a t+1).
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
# MODELOS (idénticos al paper)
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

        # Kalman causal: EM sobre training + filter sobre toda la ventana
        try:
            filtered_states = causal_kalman_filter(
                data_sub, current_train_end, test_end,
                n_iter_em=N_ITER_EM, burn_in=BURN_IN
            )
        except Exception as e:
            print(f"      ⚠ Error Kalman fold {fold_idx}: {e}")
            current_train_end += TEST_FOLD_SIZE
            continue

        # Tras burn-in, los índices se desplazan
        effective_train_end = current_train_end - BURN_IN
        effective_test_end = test_end - BURN_IN

        if effective_train_end <= 0 or effective_test_end <= effective_train_end:
            current_train_end += TEST_FOLD_SIZE
            continue

        # Scaler ajustado SOLO con training
        scaler_fold = scaler_class()
        train_states = filtered_states[:effective_train_end]
        full_states = filtered_states[:effective_test_end]

        scaler_fold.fit(train_states)
        train_scaled = scaler_fold.transform(train_states)
        full_scaled = scaler_fold.transform(full_states)

        X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
        X_all, y_all = create_sequences(full_scaled, SEQ_LENGTH)

        test_start_idx = effective_train_end - SEQ_LENGTH
        test_end_idx = effective_test_end - SEQ_LENGTH
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
            print(f"      ⚠ Error models fold {fold_idx}: {e}")

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
    print(" ROBUSTNESS ANALYSIS - KALMAN FILTER (Multivariate)")
    print(" Walk-forward | Causal Kalman FILTER | EM train-only | Train-only scaler")
    print("=" * 70)

    merged_data = pd.read_csv(DATA_PATH)
    if merged_data.isnull().sum().sum() > 0:
        merged_data = merged_data.interpolate(method='linear')
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    print(f"\nDataset: {merged_data.shape[0]} meses, {merged_data.shape[1]} variables")
    print(f"Kalman config: EM iter={N_ITER_EM}, burn-in={BURN_IN}, seq_length={SEQ_LENGTH}")

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
                        'filter': 'Kalman_causal_filter',
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
