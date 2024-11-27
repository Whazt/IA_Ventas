import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import MySQLdb
import joblib

# Función para obtener los datos
def obtener_datos():
    conexion = MySQLdb.connect(
        host="localhost",
        user="tu_usuario",         # Cambia esto
        passwd="tu_contraseña",    # Cambia esto
        db="ventas_db"             # Cambia esto
    )
    
    # Consulta para obtener las ventas
    query = """
    SELECT v.fecha, dv.total
    FROM Ventas v
    JOIN Detalles_Ventas dv ON v.id = dv.venta_id;
    """
    datos = pd.read_sql(query, conexion)
    conexion.close()

    # Procesar los datos
    datos['fecha'] = pd.to_datetime(datos['fecha'])
    datos['año'] = datos['fecha'].dt.year
    datos['mes'] = datos['fecha'].dt.month
    datos['día'] = datos['fecha'].dt.day

    return datos[['año', 'mes', 'día', 'total']]

# Preparar los datos
def preparar_datos(datos):
    X = datos[['año', 'mes', 'día']]
    y = datos[['total']]

    # Escalar los datos
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# Crear y entrenar el modelo
def entrenar_modelo(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Predicción de un único valor
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)
    return model

# Guardar el modelo y los escaladores
def guardar_modelo(model, scaler_X, scaler_y):
    model.save("modelo_ventas.h5")
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

if __name__ == "__main__":
    datos = obtener_datos()
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preparar_datos(datos)
    modelo = entrenar_modelo(X_train, y_train)
    guardar_modelo(modelo, scaler_X, scaler_y)
    print("Modelo entrenado y guardado exitosamente.")
