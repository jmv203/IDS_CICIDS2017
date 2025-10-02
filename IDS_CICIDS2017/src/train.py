# train model
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

# ==========================
# Configuración GPU
# ==========================
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detectadas:", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU configurada correctamente")

# ==========================
# Función de limpieza
# ==========================
def clean_df(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    for col in df.columns:
        if col != "Label":
            df[col] = df[col].astype('float32')
    return df

# ==========================
# Cargar dataset
# ==========================
dataset_path = "../data/MachineLearningCVE"
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
print("Archivos encontrados:", len(csv_files))

df_list = []
for file in csv_files:
    path = os.path.join(dataset_path, file)
    print("Cargando", path)
    df_list.append(pd.read_csv(path))
df = pd.concat(df_list, ignore_index=True)
print("Dataset combinado:", df.shape)

# ==========================
# Limpieza y preparación
# ==========================
df = clean_df(df)

X = df.drop("Label", axis=1).values
y = df["Label"].values

# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split train/test
X_train, X_test, y_train, y_test, y_labels_train, y_labels_test = train_test_split(
    X, y_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ==========================
# Balanceo de clases
# ==========================
classes = np.unique(y)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# ==========================
# Definir modelo
# ==========================
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(y_encoded.shape[1], activation='softmax')  # multi-clase exclusiva
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ==========================
# Callbacks
# ==========================
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
    TensorBoard(log_dir=log_dir)
]

# ==========================
# Entrenamiento
# ==========================
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ==========================
# Evaluación
# ==========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Reporte detallado
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=encoder.categories_[0]))

# ==========================
# Guardar modelo y preprocesadores
# ==========================
model.save("ids_model_improved.h5")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(encoder, "../models/encoder.pkl")
pd.DataFrame(history.history).to_csv("../models/train_history.csv", index=False)
print("Modelo y preprocesadores guardados correctamente.")
print(f"Logs de TensorBoard en: {log_dir}")
