import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="DSS Construcción - Gestión de Proyectos", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título con estilo
st.markdown("""
<h1 style='text-align: center; color: #2c3e50;'>
    🏗️ SISTEMA DE SOPORTE A LA DECISIÓN <br>
    <span style='font-size: 24px; color: #3498db;'>Gestión de Proyectos y Recursos de Construcción</span>
</h1>
""", unsafe_allow_html=True)

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
    
    # Clasificación de riesgo
    df['Nivel_Riesgo'] = pd.cut(df['Retraso'], 
                                 bins=[-float('inf'), 0, 30, float('inf')],
                                 labels=['✅ Bajo Riesgo', '⚠️ Medio Riesgo', '🔴 Alto Riesgo'])
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
    
    return modelo, scaler, le_tipo, le_clima, features

modelo, scaler, le_tipo, le_clima, features = entrenar_modelo()

# ============================================
# SIDEBAR - FILTROS
# ============================================
st.sidebar.markdown("""
<h2 style='text-align: center; color: #3498db;'>🔍 FILTROS</h2>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

tipos = st.sidebar.multiselect(
    "🏗️ Tipo de Obra",
    df['Tipo de Obra'].unique(),
    help="Selecciona uno o más tipos de obra"
)

climas = st.sidebar.multiselect(
    "☁️ Clima",
    df['Clima'].unique(),
    help="Selecciona condiciones climáticas"
)

st.sidebar.markdown("---")

# Aplicar filtros
df_filtrado = df.copy()
if tipos:
    df_filtrado = df_filtrado[df_filtrado['Tipo de Obra'].isin(tipos)]
if climas:
    df_filtrado = df_filtrado[df_filtrado['Clima'].isin(climas)]

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📊 PANEL DE CONTROL</h2>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='background-color: #3498db; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📋 TOTAL</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{}</h1>
        <p style='color: white; margin: 0;'>Proyectos</p>
    </div>
    """.format(len(df_filtrado)), unsafe_allow_html=True)

with col2:
    retraso_prom = df_filtrado['Retraso'].mean()
    color_retraso = "#27ae60" if retraso_prom <= 0 else "#e67e22" if retraso_prom <= 30 else "#e74c3c"
    st.markdown(f"""
    <div style='background-color: {color_retraso}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>⏱️ RETRASO</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{retraso_prom:.1f}</h1>
        <p style='color: white; margin: 0;'>Días promedio</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    sobrecosto_prom = df_filtrado['Desviacion_Costo'].mean()
    st.markdown(f"""
    <div style='background-color: #9b59b6; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>💰 SOBRECOSTO</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{sobrecosto_prom:.1f}%</h1>
        <p style='color: white; margin: 0;'>Promedio</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    proyectos_riesgo = len(df_filtrado[df_filtrado['Retraso'] > 30])
    st.markdown(f"""
    <div style='background-color: #e74c3c; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>⚠️ ALERTAS</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{proyectos_riesgo}</h1>
        <p style='color: white; margin: 0;'>Alto riesgo</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# GRÁFICOS MEJORADOS (SIN 3D QUE DA ERROR)
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📈 ANÁLISIS VISUAL</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🏗️ Retraso por Tipo de Obra")
    if not df_filtrado.empty:
        retraso_tipo = df_filtrado.groupby('Tipo de Obra')['Retraso'].agg(['mean', 'count']).reset_index()
        retraso_tipo.columns = ['Tipo de Obra', 'Retraso Promedio', 'Cantidad']
        
        fig1 = px.bar(
            retraso_tipo, 
            x='Tipo de Obra', 
            y='Retraso Promedio',
            color='Retraso Promedio',
            color_continuous_scale='RdYlGn_r',
            text='Cantidad',
            title='Retraso promedio por tipo de obra (con cantidad de proyectos)',
            labels={'Retraso Promedio': 'Días de retraso'}
        )
        fig1.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("##### ☁️ Distribución por Clima")
    if not df_filtrado.empty:
        clima_stats = df_filtrado.groupby('Clima').agg({
            'Retraso': 'mean',
            'Proyecto ID': 'count'
        }).reset_index()
        clima_stats.columns = ['Clima', 'Retraso Promedio', 'Cantidad']
        
        fig2 = px.pie(
            clima_stats,
            values='Cantidad',
            names='Clima',
            title='Distribución de proyectos por clima',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================
# CORRELACIÓN SIMPLIFICADA
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📊 ANÁLISIS DE CORRELACIÓN</h2>
<p style='color: #7f8c8d;'>¿Cómo afectan los recursos a los retrasos?</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🔷 Materiales vs Retraso")
    
    if not df_filtrado.empty and len(df_filtrado) > 1:
        corr_materiales = df_filtrado['Materiales'].corr(df_filtrado['Retraso'])
        
        fig3 = px.scatter(
            df_filtrado, 
            x='Materiales', 
            y='Retraso', 
            color='Tipo de Obra',
            size='Presupuesto',
            hover_data=['Proyecto ID'],
            title=f'Correlación: {corr_materiales:.2f}',
            labels={'Materiales': 'Materiales (ton)', 'Retraso': 'Retraso (días)'},
            opacity=0.7
        )
        
        # Línea de tendencia
        z = np.polyfit(df_filtrado['Materiales'], df_filtrado['Retraso'], 1)
        p = np.poly1d(z)
        fig3.add_trace(go.Scatter(
            x=df_filtrado['Materiales'].sort_values(),
            y=p(df_filtrado['Materiales'].sort_values()),
            mode='lines',
            name='Tendencia',
            line=dict(color='red', width=2)
        ))
        
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Interpretación
        if abs(corr_materiales) < 0.3:
            st.info("📌 **Interpretación:** Relación débil - Los materiales no son el factor principal")
        elif abs(corr_materiales) < 0.7:
            st.warning("📌 **Interpretación:** Relación moderada - Los materiales influyen")
        else:
            st.error("📌 **Interpretación:** Relación fuerte - Los materiales son críticos")

with col2:
    st.markdown("##### 🔶 Mano de Obra vs Retraso")
    
    if not df_filtrado.empty and len(df_filtrado) > 1:
        corr_mano_obra = df_filtrado['Mano_Obra'].corr(df_filtrado['Retraso'])
        
        fig4 = px.scatter(
            df_filtrado, 
            x='Mano_Obra', 
            y='Retraso', 
            color='Clima',
            size='Presupuesto',
            hover_data=['Proyecto ID'],
            title=f'Correlación: {corr_mano_obra:.2f}',
            labels={'Mano_Obra': 'Mano de Obra (horas)', 'Retraso': 'Retraso (días)'},
            opacity=0.7
        )
        
        # Línea de tendencia
        z = np.polyfit(df_filtrado['Mano_Obra'], df_filtrado['Retraso'], 1)
        p = np.poly1d(z)
        fig4.add_trace(go.Scatter(
            x=df_filtrado['Mano_Obra'].sort_values(),
            y=p(df_filtrado['Mano_Obra'].sort_values()),
            mode='lines',
            name='Tendencia',
            line=dict(color='red', width=2)
        ))
        
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Interpretación
        if abs(corr_mano_obra) < 0.3:
            st.info("📌 **Interpretación:** Relación débil - La mano de obra no es crítica")
        elif abs(corr_mano_obra) < 0.7:
            st.warning("📌 **Interpretación:** Relación moderada - Optimizar mano de obra ayuda")
        else:
            st.error("📌 **Interpretación:** Relación fuerte - La mano de obra es clave")

st.markdown("---")

# ============================================
# SIMULADOR DE IA MEJORADO CON OPTIMIZADOR INTELIGENTE
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>🤖 SIMULADOR DE ESCENARIOS CON IA</h2>
<p style='color: #7f8c8d;'>Configura un proyecto y descubre cómo lograr RETRASO CERO (✅ VERDE)</p>
""", unsafe_allow_html=True)

# Mostrar leyenda de colores
col_color1, col_color2, col_color3 = st.columns(3)
with col_color1:
    st.markdown("""
    <div style='background-color: #27ae60; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: white; margin: 0;'>✅ VERDE</h4>
        <p style='color: white; margin: 0;'>Retraso ≤ 0</p>
        <p style='color: white; margin: 0;'>Proyecto a tiempo</p>
    </div>
    """, unsafe_allow_html=True)
with col_color2:
    st.markdown("""
    <div style='background-color: #f39c12; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: white; margin: 0;'>🟡 AMARILLO</h4>
        <p style='color: white; margin: 0;'>0-30 días</p>
        <p style='color: white; margin: 0;'>Riesgo moderado</p>
    </div>
    """, unsafe_allow_html=True)
with col_color3:
    st.markdown("""
    <div style='background-color: #e74c3c; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: white; margin: 0;'>🔴 ROJO</h4>
        <p style='color: white; margin: 0;'> > 30 días</p>
        <p style='color: white; margin: 0;'>Alto riesgo</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# MODELO ESPECIAL PARA PROYECTOS VERDES
# ============================================
@st.cache_resource
def entrenar_modelo_verde():
    """Modelo especial que pondera más los proyectos exitosos"""
    le_tipo_v = LabelEncoder()
    le_clima_v = LabelEncoder()
    
    df_modelo = df.copy()
    df_modelo['Tipo_Cod'] = le_tipo_v.fit_transform(df_modelo['Tipo de Obra'])
    df_modelo['Clima_Cod'] = le_clima_v.fit_transform(df_modelo['Clima'])
    
    features = ['Presupuesto', 'Duracion_Estimada', 'Materiales', 'Mano_Obra', 'Tipo_Cod', 'Clima_Cod']
    X = df_modelo[features]
    y = df_modelo['Retraso']
    
    # Crear pesos: dar más importancia a proyectos exitosos (retraso <= 0)
    sample_weights = np.where(y <= 0, 3.0, 1.0)  # 3x más peso a proyectos verdes
    
    scaler_v = StandardScaler()
    X_scaled = scaler_v.fit_transform(X)
    
    modelo_v = RandomForestRegressor(
        n_estimators=100,  # Más árboles
        max_depth=8,       # Profundidad media
        min_samples_split=5,
        random_state=42
    )
    modelo_v.fit(X_scaled, y, sample_weight=sample_weights)
    
    return modelo_v, scaler_v, le_tipo_v, le_clima_v, features

# Cargar modelo especial
modelo_verde, scaler_verde, le_tipo_verde, le_clima_verde, _ = entrenar_modelo_verde()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
        <h4 style='color: #2c3e50;'>📋 PARÁMETROS DEL PROYECTO</h4>
    </div>
    """, unsafe_allow_html=True)
    
    tipo_sim = st.selectbox("🏗️ Tipo de Obra", df['Tipo de Obra'].unique(), key='tipo_sim_verde')
    clima_sim = st.selectbox("☁️ Clima", df['Clima'].unique(), key='clima_sim_verde')
    
    st.markdown("##### 💰 Recursos")
    
    presupuesto_sim = st.slider("Presupuesto ($)", 
                                min_value=1_000_000, 
                                max_value=50_000_000, 
                                value=25_000_000,
                                step=1_000_000,
                                format="$%d",
                                key='pres_sim_verde')
    
    duracion_sim = st.slider("Duración estimada (días)", 
                             min_value=50, 
                             max_value=800, 
                             value=300,
                             step=10,
                             key='dur_sim_verde')
    
    materiales_sim = st.slider("Materiales (ton)", 
                               min_value=1000, 
                               max_value=20000, 
                               value=10000,
                               step=500,
                               key='mat_sim_verde')
    
    mano_obra_sim = st.slider("Mano de obra (horas)", 
                              min_value=5000, 
                              max_value=150000,  # Aumentado el máximo
                              value=50000,
                              step=5000,
                              key='mo_sim_verde')
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🎯 SIMULAR", use_container_width=True, key='btn_sim'):
            try:
                tipo_cod = le_tipo_verde.transform([tipo_sim])[0]
                clima_cod = le_clima_verde.transform([clima_sim])[0]
                
                X_sim = np.array([[presupuesto_sim, duracion_sim, materiales_sim, 
                                  mano_obra_sim, tipo_cod, clima_cod]])
                X_sim_scaled = scaler_verde.transform(X_sim)
                
                retraso_pred = modelo_verde.predict(X_sim_scaled)[0]
                
                st.session_state['retraso_sim_verde'] = retraso_pred
                st.session_state['parametros_sim_verde'] = {
                    'tipo': tipo_sim,
                    'clima': clima_sim,
                    'presupuesto': presupuesto_sim,
                    'duracion': duracion_sim,
                    'materiales': materiales_sim,
                    'mano_obra': mano_obra_sim
                }
                
                # Generar escenarios EXTREMOS para buscar verde
                escenarios = []
                escenarios.append({'nombre': 'Actual', 'retraso': retraso_pred})
                
                # Escenarios agresivos para buscar verde
                for mo_mult in [1.3, 1.5, 1.8, 2.0, 2.5]:
                    for clima_test in ['Soleado', 'Nublado']:
                        clima_test_cod = le_clima_verde.transform([clima_test])[0]
                        X_test = np.array([[presupuesto_sim, duracion_sim, materiales_sim, 
                                           mano_obra_sim * mo_mult, tipo_cod, clima_test_cod]])
                        X_test_scaled = scaler_verde.transform(X_test)
                        ret_test = modelo_verde.predict(X_test_scaled)[0]
                        
                        nombre = f'+{int((mo_mult-1)*100)}% MO, {clima_test}'
                        escenarios.append({'nombre': nombre, 'retraso': ret_test})
                
                st.session_state['escenarios_verde'] = escenarios
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col_btn2:
        if st.button("🎯 BUSCAR VERDE (IA)", use_container_width=True, key='btn_opt'):
            with st.spinner('🔍 IA buscando combinación perfecta...'):
                try:
                    tipo_cod = le_tipo_verde.transform([tipo_sim])[0]
                    
                    mejores_params = None
                    mejor_retraso = 1000  # Iniciar con valor alto
                    
                    # Búsqueda exhaustiva
                    climas_test = ['Soleado', 'Nublado']
                    mo_multipliers = [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
                    mat_multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
                    
                    resultados = []
                    
                    for clima_test in climas_test:
                        clima_test_cod = le_clima_verde.transform([clima_test])[0]
                        
                        for mo_mult in mo_multipliers:
                            for mat_mult in mat_multipliers:
                                # Calcular valores
                                mo_test = mano_obra_sim * mo_mult
                                mat_test = materiales_sim * mat_mult
                                
                                # Limitar a rangos razonables
                                if mo_test > 200000 or mat_test > 30000:
                                    continue
                                
                                X_test = np.array([[
                                    presupuesto_sim,
                                    duracion_sim,
                                    mat_test,
                                    mo_test,
                                    tipo_cod,
                                    clima_test_cod
                                ]])
                                
                                X_test_scaled = scaler_verde.transform(X_test)
                                retraso_test = modelo_verde.predict(X_test_scaled)[0]
                                
                                resultados.append({
                                    'clima': clima_test,
                                    'mano_obra': mo_test,
                                    'materiales': mat_test,
                                    'mult_mo': mo_mult,
                                    'mult_mat': mat_mult,
                                    'retraso': retraso_test
                                })
                                
                                # Guardar el mejor (más bajo) y que sea <= 0 si es posible
                                if retraso_test < mejor_retraso:
                                    mejor_retraso = retraso_test
                                    mejores_params = {
                                        'clima': clima_test,
                                        'mano_obra': mo_test,
                                        'materiales': mat_test,
                                        'mult_mo': mo_mult,
                                        'mult_mat': mat_mult,
                                        'retraso': retraso_test
                                    }
                    
                    # Guardar resultados
                    st.session_state['resultados_busqueda'] = resultados
                    
                    if mejores_params:
                        st.session_state['optimo_encontrado_verde'] = True
                        st.session_state['mejores_params_verde'] = mejores_params
                        
                        # Mensaje según resultado
                        if mejores_params['retraso'] <= 0:
                            st.balloons()
                            st.success("🎉 ¡ENCONTRAMOS UNA COMBINACIÓN VERDE!")
                        else:
                            st.info(f"📊 Mejor retraso encontrado: {mejores_params['retraso']:.1f} días")
                    
                except Exception as e:
                    st.error(f"Error en optimización: {e}")

with col2:
    if 'retraso_sim_verde' in st.session_state:
        retraso = st.session_state['retraso_sim_verde']
        
        # Determinar color y mensaje
        if retraso <= 0:
            bg_color = "#27ae60"
            emoji = "✅"
            mensaje = "¡PROYECTO A TIEMPO!"
            st.balloons()
        elif retraso <= 30:
            bg_color = "#f39c12"
            emoji = "⚠️"
            mensaje = "RIESGO MODERADO"
        else:
            bg_color = "#e74c3c"
            emoji = "🔴"
            mensaje = "ALTO RIESGO"
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>{emoji} RESULTADO</h2>
            <h1 style='color: white; margin: 0; font-size: 72px;'>{retraso:.1f}</h1>
            <h3 style='color: white; margin: 0;'>DÍAS DE RETRASO</h3>
            <p style='color: white; margin: 10px 0 0 0; font-size: 18px;'>{mensaje}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Análisis de brecha para llegar a verde
        st.markdown("##### 📊 ANÁLISIS DE BRECHA PARA VERDE")
        
        if retraso > 0:
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.markdown(f"""
                <div style='text-align: center; background-color: #27ae60; padding: 10px; border-radius: 5px;'>
                    <p style='color: white; margin: 0; font-size: 24px;'>0</p>
                    <p style='color: white; margin: 0;'>META</p>
                </div>
                """, unsafe_allow_html=True)
            with col_g2:
                st.markdown(f"""
                <div style='text-align: center; background-color: {bg_color}; padding: 10px; border-radius: 5px;'>
                    <p style='color: white; margin: 0; font-size: 24px;'>{retraso:.1f}</p>
                    <p style='color: white; margin: 0;'>ACTUAL</p>
                </div>
                """, unsafe_allow_html=True)
            with col_g3:
                brecha = retraso
                st.markdown(f"""
                <div style='text-align: center; background-color: #3498db; padding: 10px; border-radius: 5px;'>
                    <p style='color: white; margin: 0; font-size: 24px;'>{brecha:.1f}</p>
                    <p style='color: white; margin: 0;'>BRECHA</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Cálculo de recursos necesarios
            st.markdown("##### 🔧 RECURSOS NECESARIOS PARA VERDE")
            
            # Calcular incrementos necesarios
            mo_actual = st.session_state['parametros_sim_verde']['mano_obra']
            
            # Probar diferentes incrementos
            incrementos = [1.2, 1.5, 1.8, 2.0, 2.5]
            mejor_opcion = None
            menor_retraso = retraso
            
            tipo_cod = le_tipo_verde.transform([tipo_sim])[0]
            
            for inc in incrementos:
                mo_test = mo_actual * inc
                if mo_test <= 200000:  # Límite razonable
                    clima_test_cod = le_clima_verde.transform(['Soleado'])[0]
                    X_test = np.array([[presupuesto_sim, duracion_sim, materiales_sim, 
                                       mo_test, tipo_cod, clima_test_cod]])
                    X_test_scaled = scaler_verde.transform(X_test)
                    ret_test = modelo_verde.predict(X_test_scaled)[0]
                    
                    if ret_test < menor_retraso:
                        menor_retraso = ret_test
                        mejor_opcion = (inc, ret_test)
            
            if mejor_opcion:
                inc, ret_nuevo = mejor_opcion
                st.info(f"""
                **Recomendación:** Aumenta mano de obra **{int((inc-1)*100)}%**  
                → De {mo_actual:,.0f} a {mo_actual*inc:,.0f} horas  
                → Retraso estimado: {ret_nuevo:.1f} días
                
                {"✅ ¡Alcanzarías VERDE!" if ret_nuevo <= 0 else f"🟡 Todavía en amarillo (faltan {ret_nuevo:.1f} días)"}
                """)
        
        st.markdown("---")
        
        # Mostrar resultados de optimización
        if st.session_state.get('optimo_encontrado_verde', False):
            mejor = st.session_state['mejores_params_verde']
            
            if mejor['retraso'] <= 0:
                st.success("🎯 **¡COMBINACIÓN VERDE ENCONTRADA!**")
            else:
                st.warning("📊 **MEJOR COMBINACIÓN ENCONTRADA** (casi verde)")
            
            st.markdown(f"""
            <div style='background-color: {'#27ae60' if mejor['retraso'] <= 0 else '#f39c12'}; padding: 15px; border-radius: 10px;'>
                <p style='color: white; margin: 5px;'><strong>🌤️ Clima:</strong> {mejor['clima']}</p>
                <p style='color: white; margin: 5px;'><strong>👷 Mano de obra:</strong> {mejor['mano_obra']:,.0f} horas ({mejor['mult_mo']*100:.0f}% del actual)</p>
                <p style='color: white; margin: 5px;'><strong>🏗️ Materiales:</strong> {mejor['materiales']:,.0f} ton ({mejor['mult_mat']*100:.0f}% del actual)</p>
                <p style='color: white; margin: 5px;'><strong>📊 Retraso estimado:</strong> {mejor['retraso']:.1f} días</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráficas del simulador
        if 'escenarios_verde' in st.session_state:
            st.markdown("---")
            st.markdown("##### 📊 COMPARATIVA DE ESTRATEGIAS")
            
            escenarios_df = pd.DataFrame(st.session_state['escenarios_verde'])
            
            # Ordenar por retraso
            escenarios_df = escenarios_df.sort_values('retraso')
            
            # Colores según valor
            colors = ['#27ae60' if x <= 0 else '#f39c12' if x <= 30 else '#e74c3c' 
                     for x in escenarios_df['retraso']]
            
            fig = px.bar(
                escenarios_df.head(10),  # Mostrar top 10
                x='nombre',
                y='retraso',
                title='Top 10 mejores estrategias (ordenadas por menor retraso)',
                labels={'nombre': 'Estrategia', 'retraso': 'Retraso (días)'}
            )
            
            # Colorear barras individualmente
            fig.update_traces(marker_color=colors[:10])
            
            fig.add_hline(y=0, line_dash="dash", line_color="green", 
                         annotation_text="VERDE", annotation_position="bottom right")
            fig.add_hline(y=30, line_dash="dash", line_color="red", 
                         annotation_text="LÍMITE ROJO", annotation_position="top right")
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# GUÍA ESPECIAL PARA LOGRAR VERDE
# ============================================
with st.expander("📘 GUÍA ESPECIAL: Cómo lograr VERDE (Retraso ≤ 0)", expanded=False):
    st.markdown("""
    ### 🎯 **ESTRATEGIAS COMPROBADAS PARA VERDE:**
    
    #### 1. **Tipo de obra con mejor historial:**
    - ✅ **Aeropuertos** (suelen terminar antes)
    - ✅ **Estadios** (buena planificación)
    - ✅ **Carreteras** (con MO suficiente)
    
    #### 2. **Clima óptimo:**
    - 🌞 **Soleado** (mejor rendimiento)
    - ⛅ **Nublado** (aceptable)
    - 🌧️ **Evitar:** Tormenta, Viento Fuerte
    
    #### 3. **Mano de obra (factor crítico):**
    ```
    Para proyectos PEQUEÑOS (< 200 días): > 30,000 horas
    Para proyectos MEDIANOS (200-400 días): > 60,000 horas  
    Para proyectos GRANDES (> 400 días): > 90,000 horas
    Para proyectos MUY GRANDES (> 600 días): > 120,000 horas
    ```
    
    #### 4. **Materiales optimizados:**
    - Ni muchos (causa desorden) ni pocos (causa paros)
    - Relación ideal: **0.15 - 0.25 ton/hora**
    
    #### 5. **Combinaciones ganadoras (datos históricos):**
    
    | Tipo | MO (horas) | Materiales (ton) | Clima | Resultado |
    |------|------------|------------------|-------|-----------|
    | Aeropuerto | 95,000+ | 15,000-20,000 | Soleado | ✅ Verde |
    | Carretera | 85,000+ | 10,000-15,000 | Nublado | ✅ Verde |
    | Estadio | 75,000+ | 8,000-12,000 | Soleado | ✅ Verde |
    
    ### 💡 **PRO TIP:**
    Usa el botón **"BUSCAR VERDE (IA)"** - la IA probará cientos de combinaciones
    automáticamente para encontrar la mejor para tu proyecto.
    
    ### ⚡ **ATAJO RÁPIDO:**
    Si quieres probar manualmente, empieza con:
    - **MO = 100,000 horas**
    - **Clima = Soleado**
    - **Materiales = 12,000 ton**
    - **Tipo = Aeropuerto o Carretera**
    """)
# ============================================
# ANALIZADOR DE TU PROPIO DATASET
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>🔍 ANALIZADOR DE TU DATASET REAL</h2>
<p style='color: #7f8c8d;'>Mostrando patrones de ÉXITO de TUS 100 proyectos</p>
""", unsafe_allow_html=True)

# Mostrar estadísticas reales
col_r1, col_r2, col_r3, col_r4 = st.columns(4)

with col_r1:
    verdes = len(df[df['Retraso'] <= 0])
    st.markdown(f"""
    <div style='background-color: #27ae60; padding: 15px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>{verdes}</h2>
        <p style='color: white; margin: 0;'>Proyectos VERDES</p>
        <small style='color: white;'>({verdes/100:.0%} del total)</small>
    </div>
    """, unsafe_allow_html=True)

with col_r2:
    amarillos = len(df[(df['Retraso'] > 0) & (df['Retraso'] <= 30)])
    st.markdown(f"""
    <div style='background-color: #f39c12; padding: 15px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>{amarillos}</h2>
        <p style='color: white; margin: 0;'>Proyectos AMARILLOS</p>
    </div>
    """, unsafe_allow_html=True)

with col_r3:
    rojos = len(df[df['Retraso'] > 30])
    st.markdown(f"""
    <div style='background-color: #e74c3c; padding: 15px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>{rojos}</h2>
        <p style='color: white; margin: 0;'>Proyectos ROJOS</p>
    </div>
    """, unsafe_allow_html=True)

with col_r4:
    st.markdown(f"""
    <div style='background-color: #3498db; padding: 15px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>{df['Mano_Obra'].mean():,.0f}</h2>
        <p style='color: white; margin: 0;'>MO Promedio</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MOSTRAR PATRONES DE ÉXITO DE TUS DATOS
# ============================================
st.markdown("---")
st.markdown("### 🏆 PROYECTOS VERDES EN TUS DATOS (APRENDE DE ELLOS)")

# Agrupar por tipo de obra
tipos_verdes = df[df['Retraso'] <= 0].groupby('Tipo de Obra').agg({
    'Proyecto ID': 'count',
    'Mano_Obra': 'mean',
    'Materiales': 'mean',
    'Retraso': 'mean'
}).round(0).reset_index()
tipos_verdes.columns = ['Tipo', 'Cantidad', 'MO Promedio', 'Mat Promedio', 'Retraso Prom']

st.dataframe(tipos_verdes, use_container_width=True)

# Mostrar los mejores ejemplos
st.markdown("### ⭐ TOP 10 PROYECTOS MÁS EXITOSOS (MÁS VERDES)")

top_verdes = df[df['Retraso'] <= 0].sort_values('Retraso').head(10)[
    ['Proyecto ID', 'Tipo de Obra', 'Clima', 'Mano_Obra', 'Materiales', 'Retraso']
]

for _, row in top_verdes.iterrows():
    st.markdown(f"""
    <div style='border-left: 5px solid #27ae60; padding: 10px; margin: 5px 0; background-color: #f0f9f0;'>
        <b>Proyecto {int(row['Proyecto ID'])}</b> - {row['Tipo de Obra']} en {row['Clima']}<br>
        👷 MO: {row['Mano_Obra']:,.0f} horas | 🏗️ Mat: {row['Materiales']:,.0f} ton<br>
        🎯 Retraso: {row['Retraso']:.0f} días
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SIMULADOR QUE USA SOLO TUS DATOS
# ============================================
st.markdown("---")
st.markdown("### 🎯 SIMULADOR BASADO EN TUS PROYECTOS EXITOSOS")

# Crear un selector de proyectos exitosos como referencia
proyecto_ref = st.selectbox(
    "Selecciona un proyecto exitoso como referencia:",
    options=top_verdes['Proyecto ID'].tolist(),
    format_func=lambda x: f"Proyecto {int(x)} - {df[df['Proyecto ID']==x]['Tipo de Obra'].iloc[0]} (Retraso: {df[df['Proyecto ID']==x]['Retraso'].iloc[0]:.0f} días)"
)

if proyecto_ref:
    ref_data = df[df['Proyecto ID'] == proyecto_ref].iloc[0]
    
    st.info(f"""
    **Proyecto de referencia:**
    - Tipo: {ref_data['Tipo de Obra']}
    - Clima: {ref_data['Clima']}
    - Mano de obra: {ref_data['Mano_Obra']:,.0f} horas
    - Materiales: {ref_data['Materiales']:,.0f} ton
    - Retraso: {ref_data['Retraso']:.0f} días ✅
    """)
    
    # Permitir ajustar desde ese punto de partida
    st.markdown("**Ajusta los recursos respecto al proyecto exitoso:**")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        factor_mo = st.slider("Factor Mano de Obra", 0.5, 2.0, 1.0, 0.1)
    with col_a2:
        factor_mat = st.slider("Factor Materiales", 0.5, 2.0, 1.0, 0.1)
    
    mo_nueva = ref_data['Mano_Obra'] * factor_mo
    mat_nueva = ref_data['Materiales'] * factor_mat
    
    # Predecir con el modelo
    try:
        tipo_cod = le_tipo_verde.transform([ref_data['Tipo de Obra']])[0]
        clima_cod = le_clima_verde.transform(['Soleado'])[0]  # Asumimos clima óptimo
        
        X_test = np.array([[ref_data['Presupuesto'], ref_data['Duracion_Estimada'], 
                           mat_nueva, mo_nueva, tipo_cod, clima_cod]])
        X_test_scaled = scaler_verde.transform(X_test)
        retraso_nuevo = modelo_verde.predict(X_test_scaled)[0]
        
        # Mostrar resultado
        color = '#27ae60' if retraso_nuevo <= 0 else '#f39c12' if retraso_nuevo <= 30 else '#e74c3c'
        st.markdown(f"""
        <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>RETRASO ESTIMADO</h2>
            <h1 style='color: white; margin: 0; font-size: 48px;'>{retraso_nuevo:.1f} días</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparación
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("MO vs Referencia", 
                     f"{mo_nueva:,.0f} h", 
                     f"{((mo_nueva/ref_data['Mano_Obra'])-1)*100:.0f}%")
        with col_c2:
            st.metric("Mat vs Referencia", 
                     f"{mat_nueva:,.0f} t", 
                     f"{((mat_nueva/ref_data['Materiales'])-1)*100:.0f}%")
            
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================
# TABLA DE DATOS
# ============================================
st.markdown("---")
st.markdown("""
<h2 style='color: #2c3e50;'>📋 DATOS HISTÓRICOS</h2>
""", unsafe_allow_html=True)

with st.expander("Ver todos los proyectos", expanded=False):
    # Formato condicional
    def color_retraso(val):
        if val > 30:
            return 'background-color: #ffcccc'
        elif val > 0:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #d4edda'
    
    st.dataframe(
        df_filtrado.style.applymap(color_retraso, subset=['Retraso']),
        use_container_width=True,
        height=400
    )
    
    csv = df_filtrado.to_csv(index=False)
    st.download_button(
        "📥 Descargar datos en CSV",
        csv,
        "datos_construccion.csv",
        "text/csv",
        use_container_width=True
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>🏗️ <strong>Sistema de Soporte a la Decisión - Gestión de Proyectos de Construcción</strong></p>
    <p>Modelos: Random Forest · Análisis de Correlación · Simulación de Escenarios</p>
    <p>📊 Práctica E6: Optimización de recursos y minimización de retrasos</p>
</div>
""", unsafe_allow_html=True)
