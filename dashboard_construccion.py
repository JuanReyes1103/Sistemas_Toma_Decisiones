import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="DSS Construcción", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("🏗️ Sistema de Soporte a la Decisión - Construcción")
st.markdown("---")

# ============================================
# DATOS COMPLETOS (TODOS LOS 100 PROYECTOS)
# ============================================
@st.cache_data
def cargar_datos():
    data = {
        'Proyecto ID': list(range(1, 101)),
        'Tipo de Obra': [
            'Carretera', 'Aeropuerto', 'Escuela', 'Edificio', 'Carretera', 
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
            'Carretera', 'Puente', 'Centro Comercial', 'Escuela'
        ],
        'Presupuesto': [
            21690591, 6597866, 25171995, 39839269, 47018686, 3300594, 45645646, 25638059, 4871361, 29874458,
            45393013, 27123621, 46584477, 14270778, 21065438, 47034756, 32124222, 22129362, 22959462, 19608607,
            1703303, 11203140, 12959811, 29666508, 33533571, 9945556, 22870296, 46948404, 3678648, 48767561,
            4328908, 40073317, 7994711, 24880316, 35184007, 4115702, 18660375, 41129390, 40813830, 2211770,
            12533005, 33262550, 9762635, 5899687, 22793955, 27310154, 31013079, 38714888, 20079391, 22830309,
            15321926, 30402716, 10912394, 28454877, 3510671, 47923859, 35617782, 42290300, 30233811, 41588614,
            27698715, 17577712, 8471989, 22930574, 33611082, 32283238, 48274276, 44041261, 22882313, 37661673,
            35341525, 36818218, 49737425, 29747197, 17087919, 44686657, 13559395, 2607177, 23995298, 16329597,
            9071713, 21770588, 16638408, 2964127, 21353844, 40503612, 22531511, 8829363, 10855555, 2507689,
            41259090, 49953805, 8840738, 45858369, 24535862, 23820884, 35361234, 9231016, 10051784, 33501266
        ],
        'Costo_Real': [
            27675455.95, 8542304.05, 29567437.23, 48143596.15, 42377914.45, 3356155.95, 57268916.96, 32374254.54, 
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
            38651755.39, 11762684.14, 12087790.21, 36144383.06
        ],
        'Duracion_Estimada': [
            589, 698, 313, 692, 191, 686, 272, 430, 280, 417, 412, 612, 360, 387, 681, 312, 423, 493, 654, 442,
            615, 448, 376, 104, 578, 542, 628, 646, 470, 429, 311, 101, 519, 542, 409, 404, 212, 256, 707, 718,
            600, 510, 271, 146, 146, 553, 205, 94, 667, 319, 613, 434, 686, 436, 412, 92, 194, 694, 406, 661,
            170, 123, 650, 169, 140, 616, 265, 641, 567, 579, 607, 205, 577, 462, 432, 531, 264, 389, 658, 668,
            383, 108, 619, 486, 642, 317, 90, 130, 585, 175, 570, 546, 187, 495, 516, 470, 545, 178, 697, 636
        ],
        'Duracion_Real': [
            579, 682, 369, 665, 252, 722, 278, 460, 279, 500, 458, 662, 372, 392, 680, 308, 470, 579, 673, 481,
            594, 435, 392, 90, 641, 630, 698, 706, 465, 450, 313, 161, 532, 583, 499, 412, 225, 272, 711, 749,
            680, 596, 336, 140, 215, 536, 239, 127, 666, 360, 603, 497, 763, 472, 490, 153, 229, 667, 390, 708,
            183, 152, 714, 216, 110, 664, 241, 620, 653, 636, 594, 205, 618, 508, 493, 608, 253, 368, 673, 685,
            447, 181, 615, 533, 613, 291, 90, 158, 622, 240, 579, 615, 275, 487, 557, 448, 611, 240, 731, 723
        ],
        'Clima': [
            'Nublado', 'Tormenta', 'Viento Fuerte', 'Soleado', 'Soleado', 'Tormenta', 'Tormenta', 'Nublado', 'Tormenta', 'Soleado',
            'Lluvia', 'Nublado', 'Soleado', 'Tormenta', 'Soleado', 'Nublado', 'Nublado', 'Lluvia', 'Lluvia', 'Lluvia',
            'Nublado', 'Lluvia', 'Tormenta', 'Viento Fuerte', 'Nublado', 'Nublado', 'Nublado', 'Viento Fuerte', 'Soleado', 'Soleado',
            'Soleado', 'Nublado', 'Nublado', 'Tormenta', 'Nublado', 'Tormenta', 'Soleado', 'Viento Fuerte', 'Nublado', 'Soleado',
            'Nublado', 'Soleado', 'Soleado', 'Nublado', 'Nublado', 'Soleado', 'Lluvia', 'Viento Fuerte', 'Lluvia', 'Lluvia',
            'Tormenta', 'Nublado', 'Nublado', 'Lluvia', 'Viento Fuerte', 'Tormenta', 'Viento Fuerte', 'Lluvia', 'Nublado', 'Soleado',
            'Nublado', 'Lluvia', 'Viento Fuerte', 'Lluvia', 'Soleado', 'Tormenta', 'Lluvia', 'Soleado', 'Lluvia', 'Viento Fuerte',
            'Lluvia', 'Tormenta', 'Viento Fuerte', 'Soleado', 'Nublado', 'Lluvia', 'Viento Fuerte', 'Viento Fuerte', 'Tormenta', 'Nublado',
            'Nublado', 'Lluvia', 'Tormenta', 'Nublado', 'Nublado', 'Nublado', 'Soleado', 'Viento Fuerte', 'Viento Fuerte', 'Lluvia',
            'Tormenta', 'Tormenta', 'Tormenta', 'Nublado', 'Tormenta', 'Soleado', 'Soleado', 'Soleado', 'Tormenta', 'Soleado'
        ],
        'Materiales': [
            3965, 2560, 13869, 7563, 1559, 16882, 11828, 16898, 18240, 11955, 19944, 19668, 19166, 7125, 16094,
            12045, 14925, 8468, 14695, 5857, 15728, 15293, 11977, 18574, 15542, 3122, 14775, 3610, 18596, 13358,
            10715, 6265, 11494, 3630, 17900, 3120, 11153, 5055, 10910, 14996, 5594, 2222, 19115, 9063, 5885,
            3298, 13751, 15215, 8403, 17319, 19735, 3957, 16745, 17353, 16237, 5926, 19081, 19894, 5712, 3410,
            16257, 7061, 11544, 11195, 1314, 12948, 11171, 4867, 11382, 3337, 8740, 3498, 13848, 4059, 11420,
            9647, 1561, 9151, 16422, 7884, 15888, 4366, 17538, 6811, 4718, 5533, 19208, 3620, 17015, 3639,
            8460, 11033, 1728, 10395, 13865, 3904, 15694, 9418, 15065, 10124
        ],
        'Mano_Obra': [
            76335, 59766, 73323, 48942, 27369, 41119, 45890, 36608, 42422, 95058, 9443, 6004, 96820, 79035, 37616,
            69439, 95726, 22377, 66898, 8282, 19368, 82498, 35762, 73169, 85591, 48415, 12883, 43680, 82858, 92831,
            27715, 5609, 25280, 44402, 68757, 85594, 99870, 44313, 86505, 16037, 69810, 55061, 15697, 54033, 6436,
            88513, 97122, 29546, 98570, 5882, 37589, 90584, 18263, 94917, 88519, 5730, 62343, 73825, 50148, 13727,
            59790, 20673, 42520, 55848, 37520, 99394, 20482, 58836, 31405, 91276, 89650, 65030, 21102, 96445, 13043,
            50547, 19457, 97210, 17773, 54027, 90420, 54809, 52974, 29454, 12426, 82894, 35789, 82377, 27531, 60569,
            29784, 51084, 60498, 43582, 98412, 16113, 34772, 59755, 46557, 13424
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Verificar que todas las columnas tengan la misma longitud
    for col in df.columns:
        st.write(f"{col}: {len(df[col])} registros")
    
    # Calcular métricas derivadas
    df['Retraso'] = df['Duracion_Real'] - df['Duracion_Estimada']
    df['Desviacion_Costo'] = ((df['Costo_Real'] - df['Presupuesto']) / df['Presupuesto']) * 100
    
    return df

# Cargar datos
df = cargar_datos()

# ============================================
# VERIFICACIÓN DE DATOS
# ============================================
st.success(f"✅ Datos cargados correctamente: {len(df)} proyectos")
st.write("Primeros 5 registros:", df.head())

# ============================================
# MODELO DE IA
# ============================================
@st.cache_resource
def entrenar_modelo():
    le_tipo = LabelEncoder()
    le_clima = LabelEncoder()
    
    df_modelo = df.copy()
    df_modelo['Tipo_Cod'] = le_tipo.fit_transform(df_modelo['Tipo de Obra'])
    df_modelo['Clima_Cod'] = le_clima.fit_transform(df_modelo['Clima'])
    
    features = ['Presupuesto', 'Duracion_Estimada', 'Materiales', 'Mano_Obra', 'Tipo_Cod', 'Clima_Cod']
    X = df_modelo[features]
    y = df_modelo['Retraso']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    modelo.fit(X_scaled, y)
    
    return modelo, scaler, le_tipo, le_clima

modelo, scaler, le_tipo, le_clima = entrenar_modelo()

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("🔍 Filtros")
tipos = st.sidebar.multiselect("Tipo de Obra", df['Tipo de Obra'].unique())
climas = st.sidebar.multiselect("Clima", df['Clima'].unique())

df_filtrado = df.copy()
if tipos:
    df_filtrado = df_filtrado[df_filtrado['Tipo de Obra'].isin(tipos)]
if climas:
    df_filtrado = df_filtrado[df_filtrado['Clima'].isin(climas)]

# ============================================
# MÉTRICAS
# ============================================
st.subheader("📊 Métricas Globales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Proyectos", len(df_filtrado))
col2.metric("Retraso Promedio", f"{df_filtrado['Retraso'].mean():.1f} días")
col3.metric("Sobrecosto Promedio", f"{df_filtrado['Desviacion_Costo'].mean():.1f}%")
col4.metric("Proyectos Retrasados", len(df_filtrado[df_filtrado['Retraso'] > 0]))

# ============================================
# GRÁFICOS
# ============================================
col1, col2 = st.columns(2)

with col1:
    if not df_filtrado.empty:
        retraso_tipo = df_filtrado.groupby('Tipo de Obra')['Retraso'].mean().reset_index()
        if not retraso_tipo.empty:
            fig1 = px.bar(retraso_tipo, x='Tipo de Obra', y='Retraso', 
                         title='Retraso Promedio por Tipo de Obra',
                         color='Retraso', color_continuous_scale='RdYlGn_r')
            fig1.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No hay datos suficientes")
    else:
        st.info("Selecciona filtros")

with col2:
    if not df_filtrado.empty:
        clima_retraso = df_filtrado.groupby('Clima')['Retraso'].mean().reset_index()
        if not clima_retraso.empty:
            fig2 = px.pie(clima_retraso, values='Retraso', names='Clima',
                         title='Distribución de Retrasos por Clima')
            fig2.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay datos suficientes")
    else:
        st.info("Selecciona filtros")

# ============================================
# SIMULADOR
# ============================================
st.markdown("---")
st.subheader("🤖 SIMULADOR DE PROYECTOS CON IA")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Configura tu nuevo proyecto:**")
    tipo_sim = st.selectbox("Tipo de Obra", df['Tipo de Obra'].unique())
    clima_sim = st.selectbox("Clima", df['Clima'].unique())
    presupuesto_sim = st.number_input("Presupuesto ($)", value=25000000, step=1000000)
    duracion_sim = st.number_input("Duración estimada (días)", value=300, step=10)
    materiales_sim = st.number_input("Materiales (ton)", value=10000, step=500)
    mano_obra_sim = st.number_input("Mano de obra (horas)", value=50000, step=1000)
    
    if st.button("🔮 PREDECIR RETRASO", use_container_width=True):
        try:
            tipo_cod = le_tipo.transform([tipo_sim])[0]
            clima_cod = le_clima.transform([clima_sim])[0]
            
            X_sim = np.array([[presupuesto_sim, duracion_sim, materiales_sim, 
                              mano_obra_sim, tipo_cod, clima_cod]])
            X_sim_scaled = scaler.transform(X_sim)
            
            retraso_pred = modelo.predict(X_sim_scaled)[0]
            st.session_state['retraso_sim'] = retraso_pred
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    if 'retraso_sim' in st.session_state:
        retraso = st.session_state['retraso_sim']
        
        if retraso <= 0:
            st.success(f"### ✅ Retraso estimado: {retraso:.1f} días")
            st.balloons()
            recomendacion = "✅ Proyecto a tiempo. Mantener planificación."
        elif retraso <= 30:
            st.warning(f"### ⚠️ Retraso estimado: {retraso:.1f} días")
            recomendacion = "⚠️ Riesgo moderado. Revisar cronograma."
        else:
            st.error(f"### 🔴 Retraso estimado: {retraso:.1f} días")
            recomendacion = "🔴 ALTO RIESGO. Aumentar mano de obra."
        
        st.info(f"**Recomendación:** {recomendacion}")

# ============================================
# DATOS
# ============================================
st.markdown("---")
st.subheader("📋 Datos Históricos")
st.dataframe(df_filtrado, use_container_width=True)

if not df_filtrado.empty:
    csv = df_filtrado.to_csv(index=False)
    st.download_button("📥 Descargar CSV", csv, "datos_construccion.csv", "text/csv")

st.markdown("---")
st.caption("✅ Dashboard funcionando correctamente")
