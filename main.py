from fastapi import FastAPI, HTTPException, Query
from typing import List
import os
import joblib
import glob
import pandas as pd

# Función para cargar modelo
def cargar_modelo_por_pais(pais, ruta="modelos_guardados"):
    pais_formateado = pais.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    patron = os.path.join(ruta, f"{pais_formateado}_*.joblib")
    
    archivos = glob.glob(patron)
    if not archivos:
        raise FileNotFoundError(f"No se encontró ningún modelo para {pais}")
    
    archivo_modelo = archivos[0]
    nombre_archivo = os.path.basename(archivo_modelo)
    nombre_modelo = nombre_archivo.split('_')[-1].replace('.joblib', '')
    modelo = joblib.load(archivo_modelo)
    
    return nombre_modelo, modelo

# Función para predecir
def Prediccion_Consumo(pais, listaFechas):
    nombre_modelo, model = cargar_modelo_por_pais(pais)

    if nombre_modelo == 'ARIMA':
        predicciones = model.predict(start=str(listaFechas[0]), end=str(listaFechas[-1]), typ='levels')
        return {'fechas': listaFechas, 'predicción': list(predicciones.values.tolist())}

    if nombre_modelo == 'Prophet':
        futuro_prophet = model.make_future_dataframe(periods=10, freq='Y')
        predicciones_prophet = model.predict(futuro_prophet)
        y_pred_prophet = list(predicciones_prophet[
            predicciones_prophet['ds'].dt.year.isin(listaFechas)
        ]['yhat'].values)
        return {'fechas': listaFechas, 'predicción': y_pred_prophet}

    if nombre_modelo == 'XGBoost':
        X_nuevos = pd.DataFrame({'Year_num': listaFechas})
        y_pred_nuevos = model.predict(X_nuevos)
        return {'fechas': listaFechas, 'predicción': list(y_pred_nuevos)}

    raise ValueError("Modelo no soportado")

# Crear la app FastAPI
app = FastAPI()

# Nuevo endpoint: GET con query params
@app.get("/predecir")
def predecir(pais: str, fechas: str = Query(..., description="Lista de años separados por coma")):
    try:
        # Convertir string a lista de enteros
        lista_fechas = [int(x) for x in fechas.split(",")]
        resultado = Prediccion_Consumo(pais, lista_fechas)
        return resultado
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


