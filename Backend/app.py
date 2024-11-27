from flask import Flask, jsonify, request
import pandas as pd
import MySQLdb
import numpy as np
import tensorflow as tf
import pickle

# Cargar el modelo y los escaladores
model = tf.keras.models.load_model('Backend/modelo_ventas.h5')
with open('Backend/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('Backend/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

app = Flask(__name__)

def obtener_ventas():
    conexion = MySQLdb.connect(
        host="a",
        user="u",
        passwd="I=",
        db="b=v"
    )
    query = """
    SELECT v.FechaVenta as fecha, (dv.cantv * p.PrecioP) as total,p.NombreProd as producto, p.CodProd AS producto_id, p.CategoriaProd as categoria
    FROM Ventas v
    JOIN Det_Ventas dv ON v.Id_Venta = dv.Id_Venta
    JOIN Productos p ON dv.CodProd = p.CodProd;
    """
    datos = pd.read_sql(query, conexion)
    conexion.close()
    return datos

@app.route('/informe', methods=['POST'])
def generar_informe():
    # Obtener los datos de las fechas a predecir desde el JSON del cuerpo de la solicitud
    fechas = request.json.get('fechas', [])

    # Obtener las ventas de la base de datos
    datos = obtener_ventas()
    datos['fecha'] = pd.to_datetime(datos['fecha'])
    datos['año'] = datos['fecha'].dt.year
    datos['mes'] = datos['fecha'].dt.month
    datos['día'] = datos['fecha'].dt.day

    # Ventas totales y promedio
    ventas_totales = datos.groupby(['año', 'mes']).agg({'total': 'sum'}).reset_index()
    promedio_ventas = datos.groupby(['año', 'mes']).agg({'total': 'mean'}).reset_index()

    # Producto más vendido
    producto_mas_vendido = datos.groupby('producto').agg({'total': 'sum'}).reset_index()
    producto_mas_vendido = producto_mas_vendido.sort_values('total', ascending=False).iloc[0]

    # Categoría más vendida
    categoria_mas_vendida = datos.groupby('categoria').agg({'total': 'sum'}).reset_index()
    categoria_mas_vendida = categoria_mas_vendida.sort_values('total', ascending=False).iloc[0]

    # Distribución de ventas
    distribucion_categorias = datos.groupby('categoria').agg({'total': 'sum'}).reset_index()

    # Convertir las fechas a un formato que el modelo entienda
    fechas_input = np.array([[f['año'], f['mes'], f['día'], f['producto_id'], f['categoria']] for f in fechas])

    # Escalar las fechas con el scaler correspondiente
    fechas_scaled = scaler_X.transform(fechas_input)

    # Realizar la predicción
    predicciones_scaled = model.predict(fechas_scaled)
    predicciones = scaler_y.inverse_transform(predicciones_scaled)

    # Crear el informe
    informe = {
        "ventas_totales": ventas_totales.to_dict(orient="records"),
        "promedio_ventas": promedio_ventas.to_dict(orient="records"),
        "producto_mas_vendido": {
            "producto": producto_mas_vendido['producto'],
            "total": producto_mas_vendido['total']
        },
        "categoria_mas_vendida": {
            "categoria": categoria_mas_vendida['categoria'],
            "total": categoria_mas_vendida['total']
        },
        "distribucion_categorias": distribucion_categorias.to_dict(orient="records"),
        "predicciones": [
            {"fecha": f"{f['año']}-{f['mes']:02d}-{f['día']:02d}", "prediccion": float(p[0])}
            for f, p in zip(fechas, predicciones)
        ]
    }

    return jsonify(informe)

if __name__ == "__main__":
    app.run(debug=True)
