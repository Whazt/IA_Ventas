import pandas as pd
import numpy as np
import MySQLdb
from sklearn.preprocessing import MinMaxScaler
import pickle
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense



def obtener_ventas_para_entrenamiento():
    conexion = MySQLdb.connect(
        host="om",
        user="93a",
        passwd="IxZ",
        db="bv"
    )
    query = """
    SELECT v.FechaVenta as fecha, (dv.cantv * p.PrecioP) as total, p.CodProd AS producto_id, p.CategoriaProd as categoria
    FROM Ventas v
    JOIN Det_Ventas dv ON v.Id_Venta = dv.Id_Venta
    JOIN Productos p ON dv.CodProd = p.CodProd;
    """
    datos = pd.read_sql(query, conexion)
    conexion.close()

    # Preprocesar fechas
    datos['fecha'] = pd.to_datetime(datos['fecha'])
    datos['año'] = datos['fecha'].dt.year
    datos['mes'] = datos['fecha'].dt.month
    datos['día'] = datos['fecha'].dt.day

    # Codificar categorías
    datos['categoria'] = datos['categoria'].astype('category').cat.codes

    return datos

def preprocesar_datos(datos):
    X = datos[['año', 'mes', 'día', 'producto_id', 'categoria']]
    y = datos[['total']]

    # Escalar los datos
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y

def construir_modelo():
    modelo = Sequential()
    modelo.add(Dense(64, input_dim=5, activation='relu'))
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dense(1, activation='linear'))
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

# Entrenamiento
if __name__ == "__main__":
    datos = obtener_ventas_para_entrenamiento()
    X_scaled, y_scaled, scaler_X, scaler_y = preprocesar_datos(datos)

    modelo = construir_modelo()
    modelo.fit(X_scaled, y_scaled, epochs=100, batch_size=32, validation_split=0.2)

    modelo.save('Backend/modelo_ventas.h5')
    with open('Backend/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('Backend/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
