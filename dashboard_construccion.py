pip install streamlit pandas numpy plotly scikit-learn scipy seaborn matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="DSS Construcción - IA", layout="wide", initial_sidebar_state="expanded")

# Título principal
st.title("🏗️ Sistema de Soporte a la Decisión para Gestión de Proyectos de Construcción")
st.markdown("---")

# ============================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ============================================
@st.cache_data
def cargar_datos():
    # Crear el dataframe con los datos proporcionados
    data = {
        'Proyecto ID': range(1, 101),
        'Tipo de Obra': ['Carretera', 'Aeropuerto', 'Escuela', 'Edificio', 'Carretera', 
                         'Centro Comercial', 'Escuela', 'Escuela', 'Carretera', 'Carretera',
                         'Aeropuerto', 'Estadio', 'Carretera', 'Estadio', 'Puente',
                         'Centro Comercial', 'Carretera', 'Estadio', 'Edificio', 'Escuela',
                         'Aeropuerto', 'Centro Comercial', 'Carretera', 'Hospital', 'Estadio',
                         'Escuela', 'Escuela', 'Centro Comercial', 'Centro Comercial', 'Carretera',
                         'Puente', 'Hospital', 'Estadio', 'Hospital', 'Edificio',
                         'Carretera', 'Hospital', 'Centro Comercial', 'Hospital', 'Estadio',
                         'Hospital', 'Escuela', 'Escuela', 'Estadio', 'Puente',
                         'Puente', 'Centro Comercial', 'Edificio', 'Estadio', 'Puente',
                         'Escuela', 'Aeropuerto', 'Hospital', 'Puente', 'Centro Comercial',
                         'Carretera', 'Aeropuerto', 'Carretera', 'Hospital', 'Puente',
                         'Puente', 'Hospital', 'Puente', 'Edificio', 'Estadio',
                         'Centro Comercial', 'Carretera', 'Escuela', 'Carretera', 'Hospital',
                         'Hospital', 'Aeropuerto', 'Edificio', 'Edificio', 'Puente',
                         'Escuela', 'Estadio', 'Aeropuerto', 'Centro Comercial', 'Carretera',
                         'Edificio', 'Puente', 'Hospital', 'Aeropuerto', 'Escuela',
                         'Escuela', 'Centro Comercial', 'Hospital', 'Estadio', 'Centro Comercial',
                         'Puente', 'Edificio', 'Puente', 'Escuela', 'Centro Comercial',
                         'Carretera', 'Puente', 'Centro Comercial', 'Escuela'],
        'Presupuesto ($)': [21690591, 6597866, 25171995, 39839269, 47018686, 3300594, 45645646, 25638059, 4871361, 29874458,
                           45393013, 27123621, 46584477, 14270778, 21065438, 47034756, 32124222, 22129362, 22959462, 19608607,
                           1703303, 11203140, 12959811, 29666508, 33533571, 9945556, 22870296, 46948404, 3678648, 48767561,
                           4328908, 40073317, 7994711, 24880316, 35184007, 4115702, 18660375, 41129390, 40813830, 2211770,
                           12533005, 33262550, 9762635, 5899687, 22793955, 27310154, 31013079, 38714888, 20079391, 22830309,
                           15321926, 30402716, 10912394, 28454877, 3510671, 47923859, 35617782, 42290300, 30233811, 41588614,
                           27698715, 17577712, 8471989, 22930574, 33611082, 32283238, 48274276, 44041261, 22882313, 37661673,
                           35341525, 36818218, 49737425, 29747197, 17087919, 44686657, 13559395, 2607177, 23995298, 16329597,
                           9071713, 21770588, 16638408, 2964127, 21353844, 40503612, 22531511, 8829363, 10855555, 2507689,
                           41259090, 49953805, 8840738, 45858369, 24535862, 23820884, 35361234, 9231016, 10051784, 33501266],
        'Costo Real ($)': [27675455.95, 8542304.05, 29567437.23, 48143596.15, 42377914.45, 3356155.95, 57268916.96, 32374254.54, 
                          4550254.39, 35179307.90, 53974573.39, 27355991.36, 47762710.23, 15972460.46, 25719266.33, 53921212.87,
                          33770034.89, 27566012.69, 28225633.21, 22864025.53, 1923283.38, 11223443.19, 11744343.90, 27867203.97,
                          33131908.57, 12048088.42, 22580283.93, 46867182.83, 4220734.84, 48515958.55, 5023735.99, 44621308.97,
                          8024405.93, 29163464.45, 33337710.44, 5074923.99, 22754257.14, 46242938.47, 39174763.74, 2627686.15,
                          13096821.95, 36100957.71, 11965200.81, 6010283.46, 24324501.00, 35148842.71, 35653204.72, 44997057.24,
                          18472762.52, 20934930.67, 19025059.23, 38347287.26, 11281510.72, 34471440.45, 4409442.68, 44189827.82,
                          39369310.49, 45785237.77, 34735834.84, 42614054.54, 30925231.93, 17211774.64, 9672462.21, 29551296.58,
                          31659556.27, 37636514.07, 47855860.65, 49531078.15, 23287591.10, 43464529.06, 39491296.61, 39527154.97,
                          55791537.36, 35729453.36, 17002988.75, 57517469.76, 16826908.36, 2998587.76, 29983278.70, 16142023.74,
                          11093428.25, 25639822.73, 15709118.81, 3020320.98, 25617292.43, 40539862.97, 27894701.39, 11467247.95,
                          10642900.89, 3169153.52, 41914507.35, 50576477.30, 9503900.30, 54077437.32, 22254853.62, 24800474.49,
                          38651755.39, 11762684.14, 12087790.21, 36144383.06],
        'Duración Estimada (días)': [589, 698, 313, 692, 191, 686, 272, 430, 280, 417, 412, 612, 360, 387, 681, 312, 423, 493, 654, 442,
                                     615, 448, 376, 104, 578, 542, 628, 646, 470, 429, 311, 101, 519, 542, 409, 404, 212, 256, 707, 718,
                                     600, 510, 271, 146, 146, 553, 205, 94, 667, 319, 613, 434, 686, 436, 412, 92, 194, 694, 406, 661,
                                     170, 123, 650, 169, 140, 616, 265, 641, 567, 579, 607, 205, 577, 462, 432, 531, 264, 389, 658, 668,
                                     383, 108, 619, 486, 642, 317, 90, 130, 585, 175, 570, 546, 187, 495, 516, 470, 545, 178, 697, 636],
        'Duración Real (días)': [579, 682, 369, 665, 252, 722, 278, 460, 279, 500, 458, 662, 372, 392, 680, 308, 470, 579, 673, 481,
                                 594, 435, 392, 90, 641, 630, 698, 706, 465, 450, 313, 161, 532, 583, 499, 412, 225, 272, 711, 749,
                                 680, 596, 336, 140, 215, 536, 239, 127, 666, 360, 603, 497, 763, 472, 490, 153, 229, 667, 390, 708,
                                 183, 152, 714, 216, 110, 664, 241, 620, 653, 636, 594, 205, 618, 508, 493, 608, 253, 368, 673, 685,
                                 447, 181, 615, 533, 613, 291, 90, 158, 622, 240, 579, 615, 275, 487, 557, 448, 611, 240, 731, 723],
        'Clima': ['Nublado', 'Tormenta', 'Viento Fuerte', 'Soleado', 'Soleado', 'Tormenta', 'Tormenta', 'Nublado', 'Tormenta', 'Soleado',
                  'Lluvia', 'Nublado', 'Soleado', 'Tormenta', 'Soleado', 'Nublado', 'Nublado', 'Lluvia', 'Lluvia', 'Lluvia',
                  'Nublado', 'Lluvia', 'Tormenta', 'Viento Fuerte', 'Nublado', 'Nublado', 'Nublado', 'Viento Fuerte', 'Soleado', 'Soleado',
                  'Soleado', 'Nublado', 'Nublado', 'Tormenta', 'Nublado', 'Tormenta', 'Soleado', 'Viento Fuerte', 'Nublado', 'Soleado',
                  'Nublado', 'Soleado', 'Soleado', 'Nublado', 'Nublado', 'Soleado', 'Lluvia', 'Viento Fuerte', 'Lluvia', 'Lluvia',
                  'Tormenta', 'Nublado', 'Nublado', 'Lluvia', 'Viento Fuerte', 'Tormenta', 'Viento Fuerte', 'Lluvia', 'Nublado', 'Soleado',
                  'Nublado', 'Lluvia', 'Viento Fuerte', 'Lluvia', 'Soleado', 'Tormenta', 'Lluvia', 'Soleado', 'Lluvia', 'Viento Fuerte',
                  'Lluvia', 'Tormenta', 'Viento Fuerte', 'Soleado', 'Nublado', 'Lluvia', 'Viento Fuerte', 'Viento Fuerte', 'Tormenta', 'Nublado',
                  'Nublado', 'Lluvia', 'Tormenta', 'Nublado', 'Nublado', 'Nublado', 'Soleado', 'Viento Fuerte', 'Viento Fuerte', 'Lluvia',
                  'Tormenta', 'Tormenta', 'Tormenta', 'Nublado', 'Tormenta', 'Soleado', 'Soleado', 'Soleado', 'Tormenta', 'Soleado'],
        'Materiales Usados (ton)': [3965, 2560, 13869, 7563, 1559, 16882, 11828, 16898, 18240, 11955, 19944, 19668, 19166, 7125, 16094,
                                   12045, 14925, 8468, 14695, 5857, 15728, 15293, 11977, 18574, 15542, 3122, 14775, 3610, 18596, 13358,
                                   10715, 6265, 11494, 3630, 17900, 3120, 11153, 5055, 10910, 14996, 5594, 2222, 19115, 9063, 5885,
                                   3298, 13751, 15215, 8403, 17319, 19735, 3957, 16745, 17353, 16237, 5926, 19081, 19894, 5712, 3410,
                                   16257, 7061, 11544, 11195, 1314, 12948, 11171, 4867, 11382, 3337, 8740, 3498, 13848, 4059, 11420,
                                   9647, 1561, 9151, 16422, 7884, 15888, 4366, 17538, 6811, 4718, 5533, 19208, 3620, 17015, 3639,
                                   8460, 11033, 1728, 10395, 13865, 3904, 15694, 9418, 15065, 10124],
        'Mano de Obra (horas)': [76335, 59766, 73323, 48942, 27369, 41119, 45890, 36608, 42422, 95058, 9443, 6004, 96820, 79035, 37616,
                                 69439, 95726, 22377, 66898, 8282, 19368, 82498, 35762, 73169, 85591, 48415, 12883, 43680, 82858, 92831,
                                 27715, 5609, 25280, 44402, 68757, 85594, 99870, 44313, 86505, 16037, 69810, 55061, 15697, 54033, 6436,
                                 88513, 97122, 29546, 98570, 5882, 37589, 90584, 18263, 94917, 88519, 5730, 62343, 73825, 50148, 13727,
                                 59790, 20673, 42520, 55848, 37520, 99394, 20482, 58836, 31405, 91276, 89650, 65030, 21102, 96445, 13043,
                                 50547, 19457, 97210, 17773, 54027, 90420, 54809, 52974, 29454, 12426, 82894, 35789, 82377, 27531, 60569,
                                 29784, 51084, 60498, 43582, 98412, 16113, 34772, 59755, 46557, 13424]
    }
    
    df = pd.DataFrame(data)
    
    # Calcular métricas derivadas
    df['Retraso (días)'] = df['Duración Real (días)'] - df['Duración Estimada (días)']
    df['Desviación Costo (%)'] = ((df['Costo Real ($)'] - df['Presupuesto ($)']) / df['Presupuesto ($)']) * 100
    df['Productividad (ton/hora)'] = df['Materiales Usados (ton)'] / df['Mano de Obra (horas)']
    df['Costo por Día Real ($)'] = df['Costo Real ($)'] / df['Duración Real (días)']
    
    # Clasificar riesgo
    condiciones = [
        (df['Retraso (días)'] <= 0),
        (df['Retraso (días)'] <= 30) & (df['Retraso (días)'] > 0),
        (df['Retraso (días)'] > 30)
    ]
    categorias = ['Bajo Riesgo', 'Medio Riesgo', 'Alto Riesgo']
    df['Nivel Riesgo'] = np.select(condiciones, categorias, default='Medio Riesgo')
    
    # Codificar Estado del Proyecto basado en retraso (para clasificación)
    df['Estado_Codificado'] = df['Retraso (días)'].apply(lambda x: 0 if x <= 0 else (1 if x <= 30 else 2))
    
    return df

df = cargar_datos()

# ============================================
# 2. MODELOS MATEMÁTICOS Y DE OPTIMIZACIÓN
# ============================================
class ModelosOptimizacion:
    """Clase que contiene los modelos matemáticos de optimización"""
    
    @staticmethod
    def optimizar_asignacion_recursos(tipo_obra, presupuesto, duracion_estimada, clima):
        """
        Modelo de Programación Lineal para optimizar asignación de recursos
        Minimizar: Costo total = Costo_Materiales * Materiales + Costo_MO * Horas_MO
        Sujeto a: Restricciones de productividad y tiempo
        """
        
        # Parámetros según tipo de obra (estimados de los datos)
        params = {
            'Carretera': {'prod_min': 0.05, 'prod_max': 0.15, 'costo_material': 1200, 'costo_mo': 45},
            'Aeropuerto': {'prod_min': 0.03, 'prod_max': 0.10, 'costo_material': 2500, 'costo_mo': 60},
            'Escuela': {'prod_min': 0.10, 'prod_max': 0.30, 'costo_material': 800, 'costo_mo': 35},
            'Hospital': {'prod_min': 0.08, 'prod_max': 0.20, 'costo_material': 1800, 'costo_mo': 55},
            'Edificio': {'prod_min': 0.12, 'prod_max': 0.25, 'costo_material': 1500, 'costo_mo': 40},
            'Puente': {'prod_min': 0.04, 'prod_max': 0.12, 'costo_material': 3000, 'costo_mo': 70},
            'Estadio': {'prod_min': 0.06, 'prod_max': 0.18, 'costo_material': 2000, 'costo_mo': 50},
            'Centro Comercial': {'prod_min': 0.09, 'prod_max': 0.22, 'costo_material': 1600, 'costo_mo': 48}
        }
        
        p = params.get(tipo_obra, params['Carretera'])
        
        # Ajuste por clima
        factor_clima = {'Soleado': 1.0, 'Nublado': 1.1, 'Lluvia': 1.3, 'Tormenta': 1.5, 'Viento Fuerte': 1.2}
        fc = factor_clima.get(clima, 1.2)
        
        # Función objetivo: minimizar costo total
        def costo_total(x):
            materiales, horas_mo = x
            return p['costo_material'] * materiales + p['costo_mo'] * horas_mo * fc
        
        # Restricciones
        restricciones = [
            {'type': 'ineq', 'fun': lambda x: x[0] - 1000},  # Materiales mínimos
            {'type': 'ineq', 'fun': lambda x: 50000 - x[0]},  # Materiales máximos
            {'type': 'ineq', 'fun': lambda x: x[1] - 5000},   # Horas MO mínimas
            {'type': 'ineq', 'fun': lambda x: 100000 - x[1]}, # Horas MO máximas
            {'type': 'ineq', 'fun': lambda x: (x[0] / x[1]) - p['prod_min']}, # Productividad mínima
            {'type': 'ineq', 'fun': lambda x: p['prod_max'] - (x[0] / x[1])}  # Productividad máxima
        ]
        
        # Punto inicial
        x0 = [10000, 50000]
        
        # Optimizar
        resultado = minimize(costo_total, x0, method='SLSQP', constraints=restricciones)
        
        return resultado

    @staticmethod
    def modelo_inventario(materiales_promedio, demanda_diaria, costo_pedido=5000, costo_almacenamiento=100):
        """
        Modelo EOQ (Economic Order Quantity) para gestión de inventarios
        """
        # Cantidad económica de pedido
        eoq = np.sqrt((2 * demanda_diaria * 365 * costo_pedido) / costo_almacenamiento)
        
        # Punto de reorden (asumiendo 7 días de lead time)
        punto_reorden = demanda_diaria * 7
        
        # Inventario de seguridad (asumiendo demanda variable)
        inventario_seguridad = demanda_diaria * 3
        
        return {
            'EOQ (ton)': eoq,
            'Punto de Reorden (ton)': punto_reorden,
            'Inventario Seguridad (ton)': inventario_seguridad,
            'Costo Total Anual': (demanda_diaria * 365 / eoq) * costo_pedido + (eoq/2) * costo_almacenamiento
        }

# ============================================
# 3. MODELOS DE INTELIGENCIA ARTIFICIAL
# ============================================
class ModelosIA:
    """Clase que contiene los modelos de Machine Learning"""
    
    def __init__(self, df):
        self.df = df
        self.modelo_regresion = None
        self.modelo_clasificacion = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.entrenar_modelos()
    
    def preparar_datos(self):
        """Prepara los datos para los modelos"""
        # Crear copia para evitar warnings
        datos = self.df.copy()
        
        # Codificar variables categóricas
        for col in ['Tipo de Obra', 'Clima']:
            self.label_encoders[col] = LabelEncoder()
            datos[col + '_Cod'] = self.label_encoders[col].fit_transform(datos[col])
        
        # Features para los modelos
        features = ['Presupuesto ($)', 'Duración Estimada (días)', 
                   'Materiales Usados (ton)', 'Mano de Obra (horas)',
                   'Tipo de Obra_Cod', 'Clima_Cod']
        
        X = datos[features]
        y_reg = datos['Retraso (días)']  # Para regresión
        y_clf = datos['Estado_Codificado']  # Para clasificación
        
        return X, y_reg, y_clf, features
    
    def entrenar_modelos(self):
        """Entrena los modelos de regresión y clasificación"""
        X, y_reg, y_clf, features = self.preparar_datos()
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            X_scaled, y_reg, test_size=0.2, random_state=42
        )
        
        # 1. Modelo de Regresión para predecir retrasos
        self.modelo_regresion = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        self.modelo_regresion.fit(X_train, y_reg_train)
        
        # Evaluar regresión
        y_pred_reg = self.modelo_regresion.predict(X_test)
        self.r2_score = r2_score(y_reg_test, y_pred_reg)
        self.rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
        
        # 2. Modelo de Clasificación para nivel de riesgo
        X_train_c, X_test_c, y_clf_train, y_clf_test = train_test_split(
            X_scaled, y_clf, test_size=0.2, random_state=42
        )
        
        self.modelo_clasificacion = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.modelo_clasificacion.fit(X_train_c, y_clf_train)
        
        # Evaluar clasificación
        y_pred_clf = self.modelo_clasificacion.predict(X_test_c)
        self.accuracy = accuracy_score(y_clf_test, y_pred_clf)
        
        # Importancia de features
        self.feature_importance = pd.DataFrame({
            'Feature': features,
            'Importancia': self.modelo_regresion.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        return self
    
    def predecir_retraso(self, tipo_obra, presupuesto, duracion_estimada, 
                         materiales, mano_obra, clima):
        """Predice el retraso para un nuevo proyecto"""
        
        # Codificar variables categóricas
        tipo_cod = self.label_encoders['Tipo de Obra'].transform([tipo_obra])[0]
        clima_cod = self.label_encoders['Clima'].transform([clima])[0]
        
        # Crear array de features
        features = np.array([[
            presupuesto,
            duracion_estimada,
            materiales,
            mano_obra,
            tipo_cod,
            clima_cod
        ]])
        
        # Escalar
        features_scaled = self.scaler.transform(features)
        
        # Predecir
        retraso_pred = self.modelo_regresion.predict(features_scaled)[0]
        
        # Predecir probabilidades de riesgo
        prob_riesgo = self.modelo_clasificacion.predict_proba(features_scaled)[0]
        
        return retraso_pred, prob_riesgo
    
    def recomendar_acciones(self, retraso_pred, tipo_obra, clima):
        """Recomienda acciones basadas en la predicción"""
        recomendaciones = []
        
        if retraso_pred > 30:
            recomendaciones.append("🔴 ALTO RIESGO: Aumentar mano de obra en 20%")
            recomendaciones.append("🔴 Implementar turnos extras")
            if clima in ['Tormenta', 'Lluvia']:
                recomendaciones.append("🔴 Invertir en protecciones climáticas (carpa, drenaje)")
        elif retraso_pred > 0:
            recomendaciones.append("🟡 RIESGO MEDIO: Optimizar logística de materiales")
            recomendaciones.append("🟡 Revisar cronograma semanalmente")
        else:
            recomendaciones.append("🟢 BAJO RIESGO: Mantener planificación actual")
            recomendaciones.append("🟢 Considerar reducir buffer de contingencia")
        
        # Recomendaciones específicas por tipo de obra
        if tipo_obra in ['Hospital', 'Aeropuerto']:
            recomendaciones.append("🏥 Priorizar mano de obra especializada")
        elif tipo_obra in ['Puente', 'Carretera']:
            recomendaciones.append("🛣️ Asegurar cadena de suministro de materiales")
        
        return recomendaciones

# ============================================
# 4. INICIALIZAR MODELOS
# ============================================
modelos_ia = ModelosIA(df)
modelos_opt = ModelosOptimizacion()

# ============================================
# 5. DASHBOARD INTERACTIVO
# ============================================

# Sidebar - Filtros globales
st.sidebar.header("🔍 Filtros Globales")
tipos_obra = ['Todos'] + list(df['Tipo de Obra'].unique())
tipo_seleccionado = st.sidebar.selectbox("Tipo de Obra", tipos_obra)

climas = ['Todos'] + list(df['Clima'].unique())
clima_seleccionado = st.sidebar.selectbox("Clima", climas)

# Filtrar datos
df_filtrado = df.copy()
if tipo_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Tipo de Obra'] == tipo_seleccionado]
if clima_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Clima'] == clima_seleccionado]

# Métricas principales en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Métricas del Filtro")
st.sidebar.metric("Proyectos", len(df_filtrado))
st.sidebar.metric("Retraso Promedio", f"{df_filtrado['Retraso (días)'].mean():.1f} días")
st.sidebar.metric("Precisión Modelo IA", f"{modelos_ia.accuracy*100:.1f}%")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Análisis Exploratorio", 
    "🤖 Modelos IA Predictivos", 
    "📐 Optimización Matemática",
    "🎮 Simulador de Escenarios"
])

# ============================================
# TAB 1: ANÁLISIS EXPLORATORIO
# ============================================
with tab1:
    st.header("Análisis Exploratorio de Datos")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Proyectos", len(df))
    with col2:
        retraso_prom = df['Retraso (días)'].mean()
        st.metric("Retraso Promedio", f"{retraso_prom:.1f} días")
    with col3:
        sobrecosto_prom = df['Desviación Costo (%)'].mean()
        st.metric("Sobrecosto Promedio", f"{sobrecosto_prom:.1f}%")
    with col4:
        proyectos_riesgo = len(df[df['Nivel Riesgo'] == 'Alto Riesgo'])
        st.metric("Proyectos Alto Riesgo", proyectos_riesgo)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por tipo de obra y estado
        fig = px.histogram(df, x='Tipo de Obra', color='Nivel Riesgo', 
                          title='Distribución de Proyectos por Tipo y Nivel de Riesgo',
                          barmode='group', color_discrete_map={'Bajo Riesgo':'green', 
                                                               'Medio Riesgo':'yellow', 
                                                               'Alto Riesgo':'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Impacto del clima en retrasos
        clima_retraso = df.groupby('Clima')['Retraso (días)'].agg(['mean', 'std']).reset_index()
        fig = px.bar(clima_retraso, x='Clima', y='mean', error_y='std',
                    title='Retraso Promedio por Clima (con desviación)',
                    color='Clima', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(yaxis_title="Retraso Promedio (días)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de correlación
    st.subheader("Matriz de Correlación entre Variables")
    numeric_cols = ['Presupuesto ($)', 'Costo Real ($)', 'Duración Estimada (días)', 
                   'Duración Real (días)', 'Retraso (días)', 'Materiales Usados (ton)', 
                   'Mano de Obra (horas)', 'Desviación Costo (%)']
    
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                   color_continuous_scale='RdBu_r', title="Correlaciones")
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de productividad
    st.subheader("Análisis de Productividad por Tipo de Obra")
    fig = px.box(df, x='Tipo de Obra', y='Productividad (ton/hora)', 
                color='Tipo de Obra', title='Productividad (toneladas/hora hombre)')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: MODELOS IA PREDICTIVOS
# ============================================
with tab2:
    st.header("Modelos de Inteligencia Artificial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Rendimiento del Modelo")
        st.metric("R² Score (Precisión)", f"{modelos_ia.r2_score:.3f}")
        st.metric("RMSE (Error)", f"{modelos_ia.rmse:.1f} días")
        st.metric("Accuracy Clasificación", f"{modelos_ia.accuracy*100:.1f}%")
        
        # Importancia de features
        st.subheader("🔑 Factores más Importantes")
        fig = px.bar(modelos_ia.feature_importance.head(6), 
                    x='Importancia', y='Feature', orientation='h',
                    title='Importancia de Variables en la Predicción')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Predicción vs Realidad")
        # Tomar una muestra para visualizar
        X, y_reg, _, _ = modelos_ia.preparar_datos()
        X_scaled = modelos_ia.scaler.transform(X)
        y_pred = modelos_ia.modelo_regresion.predict(X_scaled)
        
        fig = px.scatter(x=y_reg, y=y_pred, 
                        labels={'x': 'Retraso Real (días)', 'y': 'Retraso Predicho (días)'},
                        title='Modelo de Regresión: Predicciones vs Valores Reales')
        
        # Línea de perfecta predicción
        max_val = max(y_reg.max(), y_pred.max())
        min_val = min(y_reg.min(), y_pred.min())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Predicción Perfecta',
                                line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de errores por tipo de obra
    st.subheader("Análisis de Errores por Tipo de Obra")
    errores = pd.DataFrame({
        'Tipo': df['Tipo de Obra'],
        'Error': y_pred - y_reg
    }).groupby('Tipo')['Error'].agg(['mean', 'std']).reset_index()
    
    fig = px.bar(errores, x='Tipo', y='mean', error_y='std',
                title='Error de Predicción Promedio por Tipo de Obra',
                color='Tipo')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: OPTIMIZACIÓN MATEMÁTICA
# ============================================
with tab3:
    st.header("Modelos de Optimización Matemática")
    
    st.subheader("📦 Optimización de Inventarios (Modelo EOQ)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Modelo de Cantidad Económica de Pedido (EOQ)**
        
        Este modelo determina la cantidad óptima de materiales a pedir para minimizar costos totales de inventario.
        """)
        
        # Inputs para EOQ
        demanda_diaria = st.number_input("Demanda diaria de materiales (ton)", 
                                        min_value=10, max_value=500, value=100)
        costo_pedido = st.number_input("Costo por pedido ($)", 
                                      min_value=1000, max_value=20000, value=5000)
        costo_almacenamiento = st.number_input("Costo de almacenamiento ($/ton/año)", 
                                              min_value=10, max_value=500, value=100)
        
        if st.button("Calcular EOQ"):
            resultado_eoq = modelos_opt.modelo_inventario(
                df['Materiales Usados (ton)'].mean(),
                demanda_diaria,
                costo_pedido,
                costo_almacenamiento
            )
            
            for k, v in resultado_eoq.items():
                st.metric(k, f"{v:,.0f}" if 'Costo' in k else f"{v:.0f}")
    
    with col2:
        st.subheader("📊 Asignación Óptima de Recursos")
        
        tipo_obra_opt = st.selectbox("Tipo de Obra", df['Tipo de Obra'].unique(), key='opt_tipo')
        presupuesto_opt = st.number_input("Presupuesto ($)", min_value=1e6, max_value=1e8, value=30e6)
        duracion_opt = st.number_input("Duración Estimada (días)", min_value=30, max_value=1000, value=300)
        clima_opt = st.selectbox("Clima", df['Clima'].unique(), key='opt_clima')
        
        if st.button("Optimizar Recursos"):
            resultado_opt = modelos_opt.optimizar_asignacion_recursos(
                tipo_obra_opt, presupuesto_opt, duracion_opt, clima_opt
            )
            
            if resultado_opt.success:
                st.success("✅ Optimización exitosa")
                materiales_opt, horas_opt = resultado_opt.x
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Materiales Óptimos (ton)", f"{materiales_opt:.0f}")
                    st.metric("Horas MO Óptimas", f"{horas_opt:.0f}")
                with col_b:
                    st.metric("Costo Total Mínimo ($)", f"{resultado_opt.fun:,.0f}")
                    st.metric("Productividad", f"{materiales_opt/horas_opt:.3f} ton/hora")
            else:
                st.error("No se pudo encontrar una solución óptima")

# ============================================
# TAB 4: SIMULADOR DE ESCENARIOS
# ============================================
with tab4:
    st.header("🎮 Simulador de Escenarios con IA")
    st.markdown("""
    Este simulador utiliza los modelos de Machine Learning entrenados para predecir el retraso de un nuevo proyecto
    y recomendar acciones para minimizar riesgos.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Parámetros del Proyecto")
        
        tipo_nuevo = st.selectbox("Tipo de Obra", df['Tipo de Obra'].unique())
        clima_nuevo = st.selectbox("Clima", df['Clima'].unique())
        
        presupuesto_nuevo = st.slider("Presupuesto ($)", 
                                      min_value=1_000_000, 
                                      max_value=50_000_000, 
                                      value=25_000_000,
                                      step=1_000_000,
                                      format="$%d")
        
        duracion_nuevo = st.slider("Duración Estimada (días)", 
                                   min_value=50, 
                                   max_value=800, 
                                   value=365,
                                   step=10)
        
        materiales_nuevo = st.slider("Materiales (ton)", 
                                     min_value=1000, 
                                     max_value=20000, 
                                     value=10000,
                                     step=500)
        
        mano_obra_nuevo = st.slider("Mano de Obra (horas)", 
                                    min_value=5000, 
                                    max_value=100000, 
                                    value=50000,
                                    step=5000)
        
        if st.button("🚀 Simular Proyecto", use_container_width=True):
            # Predicción
            retraso_pred, prob_riesgo = modelos_ia.predecir_retraso(
                tipo_nuevo, presupuesto_nuevo, duracion_nuevo,
                materiales_nuevo, mano_obra_nuevo, clima_nuevo
            )
            
            # Recomendaciones
            recomendaciones = modelos_ia.recomendar_acciones(retraso_pred, tipo_nuevo, clima_nuevo)
            
            # Guardar en session state
            st.session_state['retraso_pred'] = retraso_pred
            st.session_state['prob_riesgo'] = prob_riesgo
            st.session_state['recomendaciones'] = recomendaciones
    
    with col2:
        st.subheader("📊 Resultados de la Simulación")
        
        if 'retraso_pred' in st.session_state:
            retraso_pred = st.session_state['retraso_pred']
            prob_riesgo = st.session_state['prob_riesgo']
            
            # Mostrar predicción con color según riesgo
            if retraso_pred <= 0:
                color = "green"
                emoji = "✅"
                nivel = "BAJO RIESGO"
            elif retraso_pred <= 30:
                color = "orange"
                emoji = "⚠️"
                nivel = "RIESGO MEDIO"
            else:
                color = "red"
                emoji = "🔴"
                nivel = "ALTO RIESGO"
            
            st.markdown(f"## {emoji} Retraso Estimado: **{retraso_pred:.1f} días**")
            st.markdown(f"### Nivel: :{color}[{nivel}]")
            
            # Probabilidades de riesgo
            st.subheader("Probabilidades por Nivel de Riesgo")
            categorias = ['Bajo', 'Medio', 'Alto']
            fig = go.Figure(data=[
                go.Bar(name='Probabilidad', x=categorias, y=prob_riesgo,
                      marker_color=['green', 'orange', 'red'])
            ])
            fig.update_layout(yaxis_title="Probabilidad", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones del Sistema")
            for rec in st.session_state['recomendaciones']:
                st.markdown(rec)
            
            # Análisis de sensibilidad
            st.subheader("📈 Análisis de Sensibilidad")
            
            # Variar mano de obra
            horas_range = np.linspace(mano_obra_nuevo * 0.5, mano_obra_nuevo * 1.5, 10)
            retrasos_sensibilidad = []
            
            for horas in horas_range:
                ret, _ = modelos_ia.predecir_retraso(
                    tipo_nuevo, presupuesto_nuevo, duracion_nuevo,
                    materiales_nuevo, horas, clima_nuevo
                )
                retrasos_sensibilidad.append(ret)
            
            fig = px.line(x=horas_range, y=retrasos_sensibilidad,
                         labels={'x': 'Horas de Mano de Obra', 'y': 'Retraso Estimado (días)'},
                         title='Sensibilidad: Impacto de la Mano de Obra en Retraso')
            fig.add_hline(y=0, line_dash="dash", line_color="green", 
                         annotation_text="Límite de retraso cero")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👈 Configura los parámetros y presiona 'Simular Proyecto'")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
**Sistema de Soporte a la Decisión para Gestión de Proyectos de Construcción**  
*Modelos implementados: Random Forest Regressor/Classifier, Programación Lineal, Modelo EOQ*
""")
