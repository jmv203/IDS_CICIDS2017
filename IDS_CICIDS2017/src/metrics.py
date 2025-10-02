# metrics.py
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==========================
# Rutas
# ==========================
MODEL_PATH = "../models/ids_model_improved.h5"
SCALER_PATH = "../models/scaler.pkl"
ENCODER_PATH = "../models/encoder.pkl"
CICIDS_PATH = "../data/MachineLearningCVE"

# ==========================
# Cargar modelo y preprocesadores
# ==========================
print("[INFO] Cargando modelo y preprocesadores...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# ==========================
# Función de limpieza
# ==========================
def clean_df(df, label_col="Label"):
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    for col in df.columns:
        if col != label_col:
            df[col] = df[col].astype('float32')
    return df

# ==========================
# Leer CSVs CICIDS2017
# ==========================
csv_files = [f for f in os.listdir(CICIDS_PATH) if f.endswith(".csv")]
print(f"[INFO] Archivos CICIDS2017 encontrados: {len(csv_files)}")

all_y_true = []
all_y_pred = []

for file in csv_files:
    path = os.path.join(CICIDS_PATH, file)
    print("Cargando", path)
    df = pd.read_csv(path)
    df = clean_df(df)

    # Separar X e y
    if "Label" not in df.columns:
        raise ValueError(f"El archivo {file} no contiene la columna 'Label'")
    y_true = df["Label"].values
    X = df.drop("Label", axis=1).values

    # Escalar
    X_scaled = scaler.transform(X)

    # Predecir
    pred_probs = model.predict(X_scaled, batch_size=1024, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    y_pred = encoder.categories_[0][pred_classes]

    # Guardar para métricas
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

# ==========================
# Convertir a arrays
# ==========================
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# ==========================
# Métricas generales
# ==========================
print("\n=== Classification Report ===")
print(classification_report(all_y_true, all_y_pred))

acc = accuracy_score(all_y_true, all_y_pred)
print(f"[INFO] Accuracy total: {acc:.4f}")

# ==========================
# Matriz de confusión
# ==========================
labels_sorted = sorted(np.unique(all_y_true))
cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_sorted)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels_sorted, yticklabels=labels_sorted, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión CICIDS2017")
plt.show()

# ==========================
# Distribución de predicciones
# ==========================
counts = pd.Series(all_y_pred).value_counts()
plt.figure(figsize=(8,5))
counts.plot(kind="bar", color=["green" if "benign" in c.lower() else "red" for c in counts.index])
plt.title("Distribución de Predicciones IDS CICIDS2017")
plt.ylabel("Número de registros")
plt.xlabel("Clase predicha")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nResumen de predicciones:")
print(counts)
