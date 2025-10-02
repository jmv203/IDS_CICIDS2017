import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# Rutas de archivos
MODEL_PATH = "../models/ids_model_improved.h5"
SCALER_PATH = "../models/scaler.pkl"
ENCODER_PATH = "../models/encoder.pkl"
INPUT_FILE = "../data/output.csv"
OUTPUT_FILE = "../data/output_pred.csv"

print("[INFO] Cargando modelo y preprocesadores...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print(f"[INFO] Leyendo {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# === SOLO columnas numéricas ===
df_features = df.select_dtypes(include=[np.number]).copy()
print(f"[INFO] Columnas numéricas usadas: {len(df_features.columns)}")

# Reemplazar inf y NaN
df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)

# Asegurar tipo float32
df_features = df_features.astype("float32")

# Comprobar número de features esperadas
expected_features = model.input_shape[1]
if df_features.shape[1] != expected_features:
    print(f"[WARN] Nº de columnas ({df_features.shape[1]}) != esperado ({expected_features})")
    if df_features.shape[1] > expected_features:
        df_features = df_features.iloc[:, :expected_features]
        print(f"[INFO] Se recortaron columnas a {expected_features}")
    else:
        raise ValueError("Faltan columnas: ajusta el archivo para que coincida con el entrenamiento.")

# Escalar y predecir
X_scaled = scaler.transform(df_features.values)
pred_probs = model.predict(X_scaled)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = encoder.categories_[0][pred_classes]

# Añadir resultados al DataFrame
df["Prediccion_IDS"] = pred_labels
df["Confianza"] = np.max(pred_probs, axis=1)
df.to_csv(OUTPUT_FILE, index=False)
print(f"[INFO] Predicciones guardadas en {OUTPUT_FILE}")

# Visualización rápida
counts = df["Prediccion_IDS"].value_counts()
plt.figure(figsize=(8,5))
counts.plot(
    kind="bar",
    color=["green" if "benign" in c.lower() else "red" for c in counts.index]
)
plt.title("Predicciones IDS (Normal vs Ataques)")
plt.ylabel("Número de registros")
plt.xlabel("Clase predicha")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nResumen de predicciones:")
print(counts)
