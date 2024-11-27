from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import MySQLdb
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64

# Configuración de Flask
app = Flask(__name__)
CORS(app)

# Cargar el modelo y los escaladores
model = tf.keras.models.load_model("modelo_ventas.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Función para obtener datos de ventas desde la base de datos
def obtener_ventas():
    conexion = MySQLdb.connect(
        host="localhost",
        user="tu_usuario",         # Cambia esto
        passwd="tu_contraseña",    # Cambia esto
        db="ventas_db"             # Cambia esto
    )
    query = """
    SELECT v.fecha, dv.total
    FROM Ventas v
    JOIN Detalles_Ventas dv ON v.id = dv.venta_id;
    """
    datos = pd.read_sql(query, conexion)
    conexion.close()
    return datos

# Función para generar gráfico con Matplotlib
def generar_grafico_ventas_totales(ventas_totales):
    fig, ax = plt.subplots()
    ax.plot(ventas_totales['fecha'], ventas_totales['total'], label='Ventas Totales')
    ax.set_title('Ventas Totales por Mes')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas Totales')
    ax.legend()

    # Guardar el gráfico como imagen en memoria
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return img

# Función para generar gráfico con Plotly (Gráfico interactivo)
def generar_grafico_predicciones(predicciones):
    fig = px.line(predicciones, x='fecha', y='prediccion', title='Predicciones de Ventas')
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Predicción de Ventas')
    # Convertir el gráfico a un objeto base64 para enviarlo al frontend
    graph_html = fig.to_html(full_html=False)
    return graph_html

# Generar el informe de ventas y gráficos
@app.route('/informe', methods=['POST'])
def generar_informe():
    datos = obtener_ventas()
    datos['fecha'] = pd.to_datetime(datos['fecha'])
    datos['año'] = datos['fecha'].dt.year
    datos['mes'] = datos['fecha'].dt.month

    # Estadísticas: Total por mes/año
    total_ventas = datos.groupby(['año', 'mes']).agg({'total': 'sum'}).reset_index()

    # Promedio de ventas por mes/año
    promedio_ventas = datos.groupby(['año', 'mes']).agg({'total': 'mean'}).reset_index()

    # Recibir las fechas de predicción (desde el frontend)
    fechas = request.json.get('fechas', [])
    fechas_input = np.array([[f['año'], f['mes'], f['día']] for f in fechas])

    # Escalar las fechas y predecir las ventas
    fechas_scaled = scaler_X.transform(fechas_input)
    predicciones_scaled = model.predict(fechas_scaled)
    predicciones = scaler_y.inverse_transform(predicciones_scaled)

    # Crear el informe
    informe = {
        "ventas_totales": total_ventas.to_dict(orient="records"),
        "promedio_ventas": promedio_ventas.to_dict(orient="records"),
        "predicciones": [
            {"fecha": f"{f['año']}-{f['mes']:02d}-{f['día']:02d}", "prediccion": float(p[0])}
            for f, p in zip(fechas, predicciones)
        ]
    }

    # Generar los gráficos
    # 1. Gráfico de ventas totales (Matplotlib)
    img_ventas_totales = generar_grafico_ventas_totales(total_ventas)

    # 2. Gráfico de predicciones de ventas (Plotly)
    predicciones_df = pd.DataFrame(fechas, columns=['año', 'mes', 'día'])
    predicciones_df['prediccion'] = predicciones.flatten()
    grafico_predicciones_html = generar_grafico_predicciones(predicciones_df)

    # Devolver los resultados con gráficos
    return jsonify({
        "informe": informe,
        "grafico_ventas_totales": base64.b64encode(img_ventas_totales.getvalue()).decode('utf-8'),
        "grafico_predicciones_html": grafico_predicciones_html
    })

if __name__ == '__main__':
    app.run(debug=True)
