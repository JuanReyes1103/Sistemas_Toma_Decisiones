import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="DSS Construcción", 
    page_icon="🏗️", 
    layout="wide"
)

# Título
st.title("🏗️ Sistema de Soporte a la Decisión - Construcción")
st.markdown("---")

# ============================================
# DATOS EN FORMATO CSV
# ============================================
@st.cache_data
def cargar_datos():
    csv_data = """Proyecto ID,Tipo de Obra,Presupuesto,Costo_Real,Duracion_Estimada,Duracion_Real,Clima,Materiales,Mano_Obra
1,Carretera,21690591,27675455.95,589,579,Nublado,3965,76335
2,Aeropuerto,6597866,8542304.05,698,682,Tormenta,2560,59766
3,Escuela,25171995,29567437.23,313,369,Viento Fuerte,13869,73323
4,Edificio,39839269,48143596.15,692,665,Soleado,7563,48942
5,Carretera,47018686,42377914.45,191,252,Soleado,1559,27369
6,Centro Comercial,3300594,3356155.95,686,722,Tormenta,16882,41119
7,Escuela,45645646,57268916.96,272,278,Tormenta,11828,45890
8,Escuela,25638059,32374254.54,430,460,Nublado,16898,36608
9,Carretera,4871361,4550254.39,280,279,Tormenta,18240,42422
10,Carretera,29874458,35179307.90,417,500,Soleado,11955,95058
11,Aeropuerto,45393013,53974573.39,412,458,Lluvia,19944,9443
12,Estadio,27123621,27355991.36,612,662,Nublado,19668,6004
13,Carretera,46584477,47762710.23,360,372,Soleado,19166,96820
14,Estadio,14270778,15972460.46,387,392,Tormenta,7125,79035
15,Puente,21065438,25719266.33,681,680,Soleado,16094,37616
16,Centro Comercial,47034756,53921212.87,312,308,Nublado,12045,69439
17,Carretera,32124222,33770034.89,423,470,Nublado,14925,95726
18,Estadio,22129362,27566012.69,493,579,Lluvia,8468,22377
19,Edificio,22959462,28225633.21,654,673,Lluvia,14695,66898
20,Escuela,19608607,22864025.53,442,481,Lluvia,5857,8282
21,Aeropuerto,1703303,1923283.38,615,594,Nublado,15728,19368
22,Centro Comercial,11203140,11223443.19,448,435,Lluvia,15293,82498
23,Carretera,12959811,11744343.90,376,392,Tormenta,11977,35762
24,Hospital,29666508,27867203.97,104,90,Viento Fuerte,18574,73169
25,Estadio,33533571,33131908.57,578,641,Nublado,15542,85591
26,Escuela,9945556,12048088.42,542,630,Nublado,3122,48415
27,Escuela,22870296,22580283.93,628,698,Nublado,14775,12883
28,Centro Comercial,46948404,46867182.83,646,706,Viento Fuerte,3610,43680
29,Centro Comercial,3678648,4220734.84,470,465,Soleado,18596,82858
30,Carretera,48767561,48515958.55,429,450,Soleado,13358,92831
31,Puente,4328908,5023735.99,311,313,Soleado,10715,27715
32,Hospital,40073317,44621308.97,101,161,Nublado,6265,5609
33,Estadio,7994711,8024405.93,519,532,Nublado,11494,25280
34,Hospital,24880316,29163464.45,542,583,Tormenta,3630,44402
35,Edificio,35184007,33337710.44,409,499,Nublado,17900,68757
36,Carretera,4115702,5074923.99,404,412,Tormenta,3120,85594
37,Hospital,18660375,22754257.14,212,225,Soleado,11153,99870
38,Centro Comercial,41129390,46242938.47,256,272,Viento Fuerte,5055,44313
39,Hospital,40813830,39174763.74,707,711,Nublado,10910,86505
40,Estadio,2211770,2627686.15,718,749,Soleado,14996,16037
41,Hospital,12533005,13096821.95,600,680,Nublado,5594,69810
42,Escuela,33262550,36100957.71,510,596,Soleado,2222,55061
43,Escuela,9762635,11965200.81,271,336,Soleado,19115,15697
44,Estadio,5899687,6010283.46,146,140,Nublado,9063,54033
45,Puente,22793955,24324501.00,146,215,Nublado,5885,6436
46,Puente,27310154,35148842.71,553,536,Soleado,3298,88513
47,Centro Comercial,31013079,35653204.72,205,239,Lluvia,13751,97122
48,Edificio,38714888,44997057.24,94,127,Viento Fuerte,15215,29546
49,Estadio,20079391,18472762.52,667,666,Lluvia,8403,98570
50,Puente,22830309,20934930.67,319,360,Lluvia,17319,5882
51,Escuela,15321926,19025059.23,613,603,Tormenta,19735,37589
52,Aeropuerto,30402716,38347287.26,434,497,Nublado,3957,90584
53,Hospital,10912394,11281510.72,686,763,Nublado,16745,18263
54,Puente,28454877,34471440.45,436,472,Lluvia,17353,94917
55,Centro Comercial,3510671,4409442.68,412,490,Viento Fuerte,16237,88519
56,Carretera,47923859,44189827.82,92,153,Tormenta,5926,5730
57,Aeropuerto,35617782,39369310.49,194,229,Viento Fuerte,19081,62343
58,Carretera,42290300,45785237.77,694,667,Lluvia,19894,73825
59,Hospital,30233811,34735834.84,406,390,Nublado,5712,50148
60,Puente,41588614,42614054.54,661,708,Soleado,3410,13727
61,Puente,27698715,30925231.93,170,183,Nublado,16257,59790
62,Hospital,17577712,17211774.64,123,152,Lluvia,7061,20673
63,Puente,8471989,9672462.21,650,714,Viento Fuerte,11544,42520
64,Edificio,22930574,29551296.58,169,216,Lluvia,11195,55848
65,Estadio,33611082,31659556.27,140,110,Soleado,1314,37520
66,Centro Comercial,32283238,37636514.07,616,664,Tormenta,12948,99394
67,Carretera,48274276,47855860.65,265,241,Lluvia,11171,20482
68,Escuela,44041261,49531078.15,641,620,Soleado,4867,58836
69,Carretera,22882313,23287591.10,567,653,Lluvia,11382,31405
70,Hospital,37661673,43464529.06,579,636,Viento Fuerte,3337,91276
71,Hospital,35341525,39491296.61,607,594,Lluvia,8740,89650
72,Aeropuerto,36818218,39527154.97,205,205,Tormenta,3498,65030
73,Edificio,49737425,55791537.36,577,618,Viento Fuerte,13848,21102
74,Edificio,29747197,35729453.36,462,508,Soleado,4059,96445
75,Puente,17087919,17002988.75,432,493,Nublado,11420,13043
76,Escuela,44686657,57517469.76,531,608,Lluvia,9647,50547
77,Estadio,13559395,16826908.36,264,253,Viento Fuerte,1561,19457
78,Aeropuerto,2607177,2998587.76,389,368,Viento Fuerte,9151,97210
79,Centro Comercial,23995298,29983278.70,658,673,Tormenta,16422,17773
80,Carretera,16329597,16142023.74,668,685,Nublado,7884,54027
81,Edificio,9071713,11093428.25,383,447,Nublado,15888,90420
82,Puente,21770588,25639822.73,108,181,Lluvia,4366,54809
83,Hospital,16638408,15709118.81,619,615,Tormenta,17538,52974
84,Aeropuerto,2964127,3020320.98,486,533,Nublado,6811,29454
85,Escuela,21353844,25617292.43,642,613,Nublado,4718,12426
86,Escuela,40503612,40539862.97,317,291,Nublado,5533,82894
87,Centro Comercial,22531511,27894701.39,90,90,Soleado,19208,35789
88,Hospital,8829363,11467247.95,130,158,Viento Fuerte,3620,82377
89,Estadio,10855555,10642900.89,585,622,Viento Fuerte,17015,27531
90,Centro Comercial,2507689,3169153.52,175,240,Lluvia,3639,60569
91,Puente,41259090,41914507.35,570,579,Tormenta,8460,29784
92,Edificio,49953805,50576477.30,546,615,Tormenta,11033,51084
93,Puente,8840738,9503900.30,187,275,Tormenta,1728,60498
94,Escuela,45858369,54077437.32,495,487,Nublado,10395,43582
95,Centro Comercial,24535862,22254853.62,516,557,Tormenta,13865,98412
96,Escuela,23820884,24800474.49,470,448,Soleado,3904,16113
97,Carretera,35361234,38651755.39,545,611,Soleado,15694,34772
98,Puente,9231016,11762684.14,178,240,Soleado,9418,59755
99,Centro Comercial,10051784,12087790.21,697,731,Tormenta,15065,46557
100,Escuela,33501266,36144383.06,636,723,Soleado,10124,13424"""
    
    df = pd.read_csv(io.StringIO(csv_data))
    df['Retraso'] = df['Duracion_Real'] - df['Duracion_Estimada']
    df['Desviacion_Costo'] = ((df['Costo_Real'] - df['Presupuesto']) / df['Presupuesto']) * 100
    return df

# Cargar datos
df = cargar_datos()

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
        fig1 = px.bar(retraso_tipo, x='Tipo de Obra', y='Retraso', 
                     title='Retraso Promedio por Tipo de Obra',
                     color='Retraso', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if not df_filtrado.empty:
        clima_retraso = df_filtrado.groupby('Clima')['Retraso'].mean().reset_index()
        fig2 = px.pie(clima_retraso, values='Retraso', names='Clima',
                     title='Distribución de Retrasos por Clima')
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# GRÁFICOS DE DISPERSIÓN (SIN TRENDLINE)
# ============================================
st.subheader("📈 Análisis de Correlación")
col3, col4 = st.columns(2)

with col3:
    fig3 = px.scatter(df_filtrado, x='Materiales', y='Retraso', 
                      color='Tipo de Obra', 
                      title='Materiales vs Retraso',
                      opacity=0.7)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.scatter(df_filtrado, x='Mano_Obra', y='Retraso', 
                      color='Clima',
                      title='Mano de Obra vs Retraso',
                      opacity=0.7)
    st.plotly_chart(fig4, use_container_width=True)

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
            st.error(f"Error en predicción: {e}")

with col2:
    if 'retraso_sim' in st.session_state:
        retraso = st.session_state['retraso_sim']
        
        if retraso <= 0:
            st.success(f"### ✅ Retraso estimado: {retraso:.1f} días")
            st.balloons()
            recomendacion = "✅ Proyecto a tiempo. Mantener planificación."
        elif retraso <= 30:
            st.warning(f"### ⚠️ Retraso estimado: {retraso:.1f} días")
            recomendacion = "⚠️ Riesgo moderado. Revisar cronograma semanalmente."
        else:
            st.error(f"### 🔴 Retraso estimado: {retraso:.1f} días")
            recomendacion = "🔴 ALTO RIESGO. Aumentar mano de obra 20%."
        
        st.info(f"**Recomendación:** {recomendacion}")

# ============================================
# TABLA DE DATOS
# ============================================
st.markdown("---")
st.subheader("📋 Datos Históricos")
st.dataframe(df_filtrado, use_container_width=True)

if not df_filtrado.empty:
    csv = df_filtrado.to_csv(index=False)
    st.download_button("📥 Descargar CSV", csv, "datos_construccion.csv", "text/csv")

st.markdown("---")
st.caption("✅ Dashboard funcionando - Versión sin dependencias extras")
