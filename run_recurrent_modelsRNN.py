"""
MODELO 10/12 — RNN (Simple Recurrent Neural Network) — COM GPU

Replica _IM_RNN.R:
    - Lê {Code}_DatasetNew.csv (features já normalizadas)
    - WFA: d1=250, d2=5, janela deslizante
    - rnn::trainr(hidden_dim=c(20,10,5), network_type="rnn",
                  sigmoid="logistic", numepochs=5, batchsize=200)
    - Dados reshape: X=(1, 250, n_features), Y=(1, 250, 1)
      Cada timestep = 1 dia com todas as features
    - Output: predição por timestep via TimeDistributed(Dense(1))
    - Modelo NOVO a cada rodada (reset de pesos)
    - Salva {Code}_TradeSignal_RNN.csv

Shape dos dados no R:
    x = t(features)               → (n_features, 250)
    X = array(x, dim=c(1, 250, n_features))  → (batch=1, time=250, feat)
    Y = array(y, dim=c(1, 250, 1))           → (batch=1, time=250, out=1)

Uso:
    python 04_model_RNN.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ===================== CONFIGURAÇÃO =====================
BASE_DIR = Path(r"C:\Users\paulo\Desktop\b3_2\B3ICS")
SEC_NAMES = BASE_DIR / ".NewB3_pruned.csv"

TRAIN_SIZE = 250
TEST_SIZE = 5

# Parâmetros do R (rnn::trainr)
HIDDEN_LAYERS = [20, 10, 5]
LEARNING_RATE = 0.01
EPOCHS = 5
BATCH_SIZE = 200
# ========================================================

import tensorflow as tf

# --- Configuração GPU ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] {len(gpus)} GPU(s) detectada(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU] Erro ao configurar GPU: {e}")
else:
    print("[CPU] Nenhuma GPU detectada. Usando CPU.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed, InputLayer
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)


def build_rnn(n_features: int) -> Sequential:
    """
    Constrói Stacked SimpleRNN.
    Equivalente a rnn::trainr(hidden_dim=c(20,10,5), network_type="rnn").
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(None, n_features)))

    for i, units in enumerate(HIDDEN_LAYERS):
        model.add(SimpleRNN(
            units=units,
            return_sequences=True,
            activation="sigmoid",   # sigmoid="logistic" no R
        ))

    # Saída: 1 predição por timestep
    model.add(TimeDistributed(Dense(1, activation="sigmoid")))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
    )
    return model


def reset_weights(model):
    """Reinicializa pesos (equivalente a modelo novo)."""
    for layer in model.layers:
        if hasattr(layer, "kernel_initializer") and hasattr(layer, "kernel"):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, "recurrent_initializer") and hasattr(layer, "recurrent_kernel"):
            layer.recurrent_kernel.assign(
                layer.recurrent_initializer(layer.recurrent_kernel.shape))
        if hasattr(layer, "bias_initializer") and hasattr(layer, "bias"):
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))
    model.optimizer = Adam(learning_rate=LEARNING_RATE)


def read_codes(path: Path) -> list:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return df["Code"].str.strip().str.upper().tolist()


def run_wfa_rnn(code: str, base_dir: Path) -> dict:
    """Executa Walk-Forward Analysis com RNN para um ticker."""
    infile = base_dir / f"{code}_DatasetNew.csv"
    outfile = base_dir / f"{code}_TradeSignal_RNN.csv"

    if outfile.exists():
        return {"Code": code, "status": "skipped", "signals": 0}

    if not infile.exists():
        return {"Code": code, "status": "no_DatasetNew", "signals": 0}

    try:
        df = pd.read_csv(infile, encoding="utf-8-sig")
    except Exception as e:
        return {"Code": code, "status": f"read_error: {e}", "signals": 0}

    if df.shape[1] < 3:
        return {"Code": code, "status": "too_few_columns", "signals": 0}

    date_col = df.columns[0]
    label_col = df.columns[-1]
    feature_cols = df.columns[1:-1].tolist()
    n_features = len(feature_cols)

    # --- Alinhamento WFA ---
    M = len(df)
    if M < TRAIN_SIZE + TEST_SIZE:
        return {"Code": code, "status": f"too_few_rows ({M})", "signals": 0}

    Q = (M - TRAIN_SIZE) // TEST_SIZE
    H = (M - TRAIN_SIZE) - TEST_SIZE * Q
    df = df.iloc[H:].reset_index(drop=True)

    dates = df[date_col].values
    X_all = df[feature_cols].values.astype(float)
    y_all = df[label_col].values.astype(int)

    # Construir modelo UMA VEZ por ticker
    model = build_rnn(n_features)

    predict_signal = []
    predict_dates = []

    # --- Loop WFA ---
    for i in range(Q):
        train_start = i * TEST_SIZE
        train_end = train_start + TRAIN_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE

        # Reshape para 3D: (batch=1, timesteps, features)
        # R: X = array(x, dim=c(1, 250, n_features))
        X_train = X_all[train_start:train_end].reshape(1, TRAIN_SIZE, n_features)
        y_train = y_all[train_start:train_end].reshape(1, TRAIN_SIZE, 1)
        X_test = X_all[test_start:test_end].reshape(1, TEST_SIZE, n_features)
        test_dates_i = dates[test_start:test_end]

        if len(np.unique(y_all[train_start:train_end])) < 2:
            preds = [int(y_all[train_start])] * TEST_SIZE
        else:
            try:
                # Reset pesos (modelo novo)
                reset_weights(model)

                # Treinar
                model.fit(
                    X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )

                # Predição: retorna (1, 5, 1) → flatten → threshold
                # R: Y_test = predictr(model, X_test)
                #    class_predict = ifelse(Y_test>=0.5, 1, 0)[1:5]
                pred_raw = model.predict(X_test, verbose=0)
                preds = (pred_raw.flatten() >= 0.5).astype(int).tolist()[:TEST_SIZE]
            except Exception:
                preds = [0] * TEST_SIZE

        predict_signal.extend(preds)
        predict_dates.extend(test_dates_i)

    tf.keras.backend.clear_session()

    # --- Salvar ---
    if predict_signal:
        df_out = pd.DataFrame({"Date": predict_dates, "pre_signal": predict_signal})
        df_out.to_csv(outfile, index=False, encoding="utf-8-sig")

    return {"Code": code, "status": "ok", "signals": len(predict_signal)}


def main():
    codes = read_codes(SEC_NAMES)
    print(f"Modelo: RNN (hidden={HIDDEN_LAYERS}, sigmoid, lr={LEARNING_RATE})")
    print(f"WFA: d1={TRAIN_SIZE}, d2={TEST_SIZE}")
    print(f"Epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"Tickers: {len(codes)}\n")

    report = []
    for code in tqdm(codes, desc="RNN Walk-Forward"):
        result = run_wfa_rnn(code, BASE_DIR)
        report.append(result)

    report_df = pd.DataFrame(report)
    report_df.to_csv(BASE_DIR / "model_RNN_report.csv", index=False, encoding="utf-8-sig")

    n_ok = (report_df["status"] == "ok").sum()
    n_skip = (report_df["status"] == "skipped").sum()
    avg_signals = report_df.loc[report_df["status"] == "ok", "signals"].mean()

    print(f"\n{'='*50}")
    print(f"Concluído: {n_ok} processados, {n_skip} já existiam.")
    print(f"Média de sinais por ação: {avg_signals:.0f}")
    print(f"Relatório: model_RNN_report.csv")


if __name__ == "__main__":
    main()
