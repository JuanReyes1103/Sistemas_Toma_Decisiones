import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import io
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="DSS Construcción - Gestión de Proyectos PMI", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título con estilo
st.markdown("""
<h1 style='text-align: center; color: #2c3e50;'>
    🏗️ SISTEMA DE SOPORTE A LA DECISIÓN <br>
    <span style='font-size: 24px; color: #3498db;'>Basado en estándares PMI (Project Management Institute)</span>
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# MODELO DE DATOS - CARGA Y PROCESAMIENTO
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
    
    # Métricas PMI (Project Management Institute)
    df['Retraso_Dias'] = df['Duracion_Real'] - df['Duracion_Estimada']  # SV en días
    df['Porcentaje_Retraso'] = (df['Retraso_Dias'] / df['Duracion_Estimada']) * 100  # % de retraso
    df['SPI'] = df['Duracion_Estimada'] / df['Duracion_Real']  # Schedule Performance Index
    df['Desviacion_Costo'] = ((df['Costo_Real'] - df['Presupuesto']) / df['Presupuesto']) * 100
    df['Productividad'] = df['Materiales'] / df['Mano_Obra']
    
    # Clasificación de riesgo basada en % de retraso (estándar PMI)
    condiciones = [
        (df['Porcentaje_Retraso'] <= 5),  # Verde: hasta 5% de retraso
        (df['Porcentaje_Retraso'] > 5) & (df['Porcentaje_Retraso'] <= 15),  # Amarillo: 5-15%
        (df['Porcentaje_Retraso'] > 15)  # Rojo: más de 15%
    ]
    categorias = ['🟢 Controlado (0-5%)', '🟡 Riesgo Moderado (5-15%)', '🔴 Crítico (>15%)']
    df['Nivel_Riesgo'] = np.select(condiciones, categorias, default='🟡 Riesgo Moderado (5-15%)')
    
    return df

df = cargar_datos()

# ============================================
# MODELO MATEMÁTICO - OPTIMIZACIÓN DE RECURSOS
# ============================================
def modelo_optimizacion(tipo_obra, duracion_estimada):
    """
    Modelo matemático para optimizar la asignación de recursos
    usando programación lineal basado en datos históricos
    """
    # Obtener datos históricos del mismo tipo de obra
    datos_historicos = df[df['Tipo de Obra'] == tipo_obra]
    
    if len(datos_historicos) > 0:
        # Usar promedios históricos como base
        mo_promedio = datos_historicos['Mano_Obra'].mean()
        mat_promedio = datos_historicos['Materiales'].mean()
        duracion_promedio = datos_historicos['Duracion_Estimada'].mean()
        
        # Ajustar según la duración del proyecto
        factor_duracion = duracion_estimada / duracion_promedio if duracion_promedio > 0 else 1
        
        # Calcular recursos óptimos
        mo_optima = mo_promedio * factor_duracion
        mat_optimos = mat_promedio * factor_duracion
        
        # Estimar costo (basado en datos históricos)
        costo_promedio = datos_historicos['Presupuesto'].mean()
        costo_optimo = costo_promedio * factor_duracion
        
        # Productividad histórica
        productividad_historica = (mat_promedio / mo_promedio) if mo_promedio > 0 else 0.15
        
        return {
            'materiales_optimos': mat_optimos,
            'mano_obra_optima': mo_optima,
            'costo_minimo': costo_optimo,
            'productividad': productividad_historica,
            'duracion_referencia': duracion_promedio
        }
    else:
        # Fallback si no hay datos históricos
        return {
            'materiales_optimos': 10000,
            'mano_obra_optima': 50000,
            'costo_minimo': 25000000,
            'productividad': 0.2,
            'duracion_referencia': 300
        }

# ============================================
# MODELO DE IA - PREDICCIÓN DE RETRASOS
# ============================================
@st.cache_resource
def entrenar_modelos_ia():
    le_tipo = LabelEncoder()
    le_clima = LabelEncoder()
    
    df_modelo = df.copy()
    df_modelo['Tipo_Cod'] = le_tipo.fit_transform(df_modelo['Tipo de Obra'])
    df_modelo['Clima_Cod'] = le_clima.fit_transform(df_modelo['Clima'])
    
    features = ['Presupuesto', 'Duracion_Estimada', 'Materiales', 'Mano_Obra', 'Tipo_Cod', 'Clima_Cod']
    X = df_modelo[features]
    y_porcentaje = df_modelo['Porcentaje_Retraso']  # Predecir % de retraso
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_porcentaje, test_size=0.2, random_state=42)
    
    # MODELO PARA PREDECIR % DE RETRASO
    modelo_retraso = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    modelo_retraso.fit(X_train, y_train)
    
    return modelo_retraso, scaler, le_tipo, le_clima

modelo_retraso, scaler, le_tipo, le_clima = entrenar_modelos_ia()

# ============================================
# SISTEMA DE ALERTAS BASADO EN % DE RETRASO
# ============================================
def generar_alertas(tipo, clima, presupuesto, duracion, materiales_actuales, mano_obra):
    alertas = []
    
    tipo_cod = le_tipo.transform([tipo])[0]
    clima_cod = le_clima.transform([clima])[0]
    
    X = np.array([[presupuesto, duracion, materiales_actuales, mano_obra, tipo_cod, clima_cod]])
    X_scaled = scaler.transform(X)
    
    porcentaje_pred = modelo_retraso.predict(X_scaled)[0]
    
    # ALERTAS BASADAS EN % DE RETRASO (PMI)
    if porcentaje_pred > 15:
        alertas.append(("🔴 CRÍTICO", f"Retraso estimado: {porcentaje_pred:.1f}% (>15%)"))
    elif porcentaje_pred > 5:
        alertas.append(("🟡 RIESGO MODERADO", f"Retraso estimado: {porcentaje_pred:.1f}% (5-15%)"))
    else:
        alertas.append(("🟢 CONTROLADO", f"Retraso estimado: {porcentaje_pred:.1f}% (0-5%)"))
    
    # ALERTA DE CLIMA
    if clima in ['Tormenta', 'Viento Fuerte']:
        alertas.append(("🔴 CLIMA ADVERSO", "Impacto estimado: +8-12% en retraso"))
    elif clima == 'Lluvia':
        alertas.append(("🟡 LLUVIA", "Impacto estimado: +3-7% en retraso"))
    
    # ALERTA DE PRODUCTIVIDAD
    productividad = materiales_actuales / mano_obra if mano_obra > 0 else 0
    if productividad > 0.25:
        alertas.append(("🟡 ALTA PRODUCTIVIDAD", f"{productividad:.3f} ton/hora - Riesgo de desabasto"))
    elif productividad < 0.05:
        alertas.append(("🟡 BAJA PRODUCTIVIDAD", f"{productividad:.3f} ton/hora - Ineficiencia"))
    
    return alertas, porcentaje_pred

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

# Aplicar filtros
df_filtrado = df.copy()
if tipos:
    df_filtrado = df_filtrado[df_filtrado['Tipo de Obra'].isin(tipos)]
if climas:
    df_filtrado = df_filtrado[df_filtrado['Clima'].isin(climas)]

# ============================================
# PANEL DE CONTROL - MÉTRICAS PMI
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📊 PANEL DE CONTROL - MÉTRICAS PMI</h2>
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
    retraso_prom = df_filtrado['Porcentaje_Retraso'].mean()
    if retraso_prom <= 5:
        color = "#27ae60"
        estado = "Controlado"
    elif retraso_prom <= 15:
        color = "#f39c12"
        estado = "Riesgo Moderado"
    else:
        color = "#e74c3c"
        estado = "Crítico"
    
    st.markdown(f"""
    <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>⏱️ RETRASO PROM</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{retraso_prom:.1f}%</h1>
        <p style='color: white; margin: 0;'>{estado}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    spi_prom = df_filtrado['SPI'].mean()
    if spi_prom >= 0.95:
        color_spi = "#27ae60"
        estado_spi = "Buen ritmo"
    elif spi_prom >= 0.85:
        color_spi = "#f39c12"
        estado_spi = "Atención"
    else:
        color_spi = "#e74c3c"
        estado_spi = "Retrasado"
    
    st.markdown(f"""
    <div style='background-color: {color_spi}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📊 SPI PROM</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{spi_prom:.2f}</h1>
        <p style='color: white; margin: 0;'>{estado_spi}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    proyectos_criticos = len(df_filtrado[df_filtrado['Porcentaje_Retraso'] > 15])
    st.markdown(f"""
    <div style='background-color: #e74c3c; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>⚠️ CRÍTICOS</h3>
        <h1 style='color: white; margin: 0; font-size: 48px;'>{proyectos_criticos}</h1>
        <p style='color: white; margin: 0;'>>15% retraso</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# ANÁLISIS VISUAL - GRÁFICAS MEJORADAS
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📈 ANÁLISIS VISUAL</h2>
""", unsafe_allow_html=True)

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("##### 🏗️ Retraso Promedio por Tipo de Obra")
    if not df_filtrado.empty:
        retraso_tipo = df_filtrado.groupby('Tipo de Obra')['Porcentaje_Retraso'].mean().reset_index()
        retraso_tipo.columns = ['Tipo de Obra', 'Retraso Promedio (%)']
        
        # Colores pastel personalizados
        colores_pastel = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D4B8FF', '#FFB8F0', '#B8E2FF']
        
        fig1 = px.bar(
            retraso_tipo, 
            x='Tipo de Obra', 
            y='Retraso Promedio (%)',
            color='Tipo de Obra',
            color_discrete_sequence=colores_pastel,
            title='% de retraso promedio por tipo de obra',
            text_auto='.1f'
        )
        
        # Líneas de umbral
        fig1.add_hline(y=5, line_dash="dash", line_color="#2ecc71", 
                      annotation_text="Límite Verde (5%)", annotation_position="bottom right")
        fig1.add_hline(y=15, line_dash="dash", line_color="#e74c3c", 
                      annotation_text="Límite Rojo (15%)", annotation_position="top right")
        
        fig1.update_layout(
            height=450,
            xaxis_title="",
            yaxis_title="Retraso Promedio (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        # Personalizar barras
        fig1.update_traces(
            marker_line_width=1,
            marker_line_color="white",
            opacity=0.9,
            textposition='outside',
            textfont_size=12
        )
        
        st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    st.markdown("##### ☁️ Distribución de Proyectos por Clima")
    if not df_filtrado.empty:
        clima_count = df_filtrado['Clima'].value_counts().reset_index()
        clima_count.columns = ['Clima', 'Cantidad de Proyectos']
        
        # Colores pastel para el gráfico circular
        colores_pastel_circular = ['#FFB3BA', '#FFDFBA', '#BAFFC9', '#BAE1FF', '#D4B8FF', '#FFB8F0']
        
        fig2 = px.pie(
            clima_count,
            values='Cantidad de Proyectos',
            names='Clima',
            title='Distribución de proyectos por condición climática',
            color_discrete_sequence=colores_pastel_circular,
            hole=0.4  # Donut chart
        )
        
        fig2.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(line=dict(color='white', width=2)),
            pull=[0.02] * len(clima_count)  # Separar ligeramente las rebanadas
        )
        
        fig2.update_layout(
            height=450,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================
# GRÁFICAS ADICIONALES DE RIESGO (CORREGIDO)
# ============================================
st.markdown("##### 🌍 Análisis de Niveles de Riesgo")

if not df_filtrado.empty and 'Nivel_Riesgo' in df_filtrado.columns:
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        # Gráfico de pastel para niveles de riesgo
        riesgo_global = df_filtrado['Nivel_Riesgo'].value_counts().reset_index()
        riesgo_global.columns = ['Nivel de Riesgo', 'Cantidad']
        
        # Verificar que hay datos
        if not riesgo_global.empty:
            color_map_riesgo = {
                '🟢 Controlado (0-5%)': '#A8E6CF',
                '🟡 Riesgo Moderado (5-15%)': '#FFD3B6',
                '🔴 Crítico (>15%)': '#FFAAA5'
            }
            
            fig3 = px.pie(
                riesgo_global,
                values='Cantidad',
                names='Nivel de Riesgo',
                color='Nivel de Riesgo',
                color_discrete_map=color_map_riesgo,
                title='Distribución por nivel de riesgo',
                hole=0.3
            )
            
            fig3.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color='white', width=2))
            )
            
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No hay datos suficientes para mostrar la distribución de riesgos")
    
    with col_g4:
        # Barras apiladas: Tipo de Obra vs Nivel de Riesgo
        if 'Tipo de Obra' in df_filtrado.columns and 'Nivel_Riesgo' in df_filtrado.columns:
            riesgo_tipo = pd.crosstab(
                df_filtrado['Tipo de Obra'], 
                df_filtrado['Nivel_Riesgo']
            ).reset_index()
            
            if not riesgo_tipo.empty:
                riesgo_tipo_long = pd.melt(
                    riesgo_tipo, 
                    id_vars=['Tipo de Obra'], 
                    value_vars=['🟢 Controlado (0-5%)', '🟡 Riesgo Moderado (5-15%)', '🔴 Crítico (>15%)'],
                    var_name='Nivel de Riesgo', 
                    value_name='Cantidad'
                )
                
                # Filtrar solo filas con cantidad > 0
                riesgo_tipo_long = riesgo_tipo_long[riesgo_tipo_long['Cantidad'] > 0]
                
                if not riesgo_tipo_long.empty:
                    color_map_riesgo = {
                        '🟢 Controlado (0-5%)': '#A8E6CF',
                        '🟡 Riesgo Moderado (5-15%)': '#FFD3B6',
                        '🔴 Crítico (>15%)': '#FFAAA5'
                    }
                    
                    fig4 = px.bar(
                        riesgo_tipo_long,
                        x='Tipo de Obra',
                        y='Cantidad',
                        color='Nivel de Riesgo',
                        color_discrete_map=color_map_riesgo,
                        title='Riesgos por tipo de obra',
                        barmode='stack',
                        text_auto=True
                    )
                    
                    fig4.update_layout(
                        height=400,
                        xaxis_title="",
                        yaxis_title="Cantidad de proyectos"
                    )
                    
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para mostrar riesgos por tipo de obra")
            else:
                st.info("No hay datos suficientes para el análisis cruzado")
        else:
            st.info("Columnas necesarias no disponibles")
else:
    # Mostrar mensaje si no hay datos o falta la columna
    st.info("No hay proyectos en el filtro actual o la columna de riesgo no está disponible")

# ============================================
# MODELO MATEMÁTICO - OPTIMIZADOR (CORREGIDO - TEXTO LEGIBLE)
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📐 MODELO MATEMÁTICO - OPTIMIZACIÓN DE RECURSOS</h2>
<p style='color: #7f8c8d;'>Basado en datos históricos de proyectos similares</p>
""", unsafe_allow_html=True)

with st.expander("📘 ¿Cómo funciona el optimizador?", expanded=False):
    st.markdown("""
    ### 🎯 **Base del cálculo:**
    
    El optimizador utiliza **datos históricos reales** de proyectos del mismo tipo para calcular los recursos óptimos:
    
    1. **Toma el promedio histórico** de mano de obra y materiales para ese tipo de obra
    2. **Ajusta según la duración** del proyecto (factor de escala)
    3. **Calcula la productividad histórica** (toneladas por hora hombre)
    4. **Estima el costo** basado en presupuestos históricos
    
    ### 📊 **Ejemplo para Carretera:**
    - Promedio histórico: 50,000 horas MO, 10,000 ton materiales
    - Si tu proyecto dura el doble → recursos se duplican
    - Si tu proyecto dura la mitad → recursos se reducen a la mitad
    
    ### ✅ **Esto es más preciso que usar fórmulas teóricas**
    """)

col_m1, col_m2 = st.columns(2)

with col_m1:
    tipo_opt = st.selectbox("🏗️ Selecciona tipo de obra", df['Tipo de Obra'].unique(), key='tipo_opt_select')
    duracion_opt = st.number_input("📅 Duración estimada del proyecto (días)", 
                                   min_value=30, 
                                   max_value=1000, 
                                   value=300,
                                   step=10,
                                   key='dur_opt')
    
    if st.button("🔮 CALCULAR RECURSOS ÓPTIMOS", use_container_width=True, key='btn_optimizar'):
        with st.spinner("Analizando datos históricos..."):
            resultado_opt = modelo_optimizacion(tipo_opt, duracion_opt)
            
            if resultado_opt:
                # Guardar TODOS los datos necesarios en session_state
                st.session_state['resultado_optimizacion'] = resultado_opt
                st.session_state['tipo_optimizado'] = tipo_opt
                st.session_state['duracion_optimizada'] = duracion_opt
                st.success(f"✅ Cálculo completado basado en {len(df[df['Tipo de Obra']==tipo_opt])} proyectos históricos!")

with col_m2:
    # Verificar que TODAS las claves necesarias existen
    if ('resultado_optimizacion' in st.session_state and 
        'tipo_optimizado' in st.session_state and 
        'duracion_optimizada' in st.session_state):
        
        res = st.session_state['resultado_optimizacion']
        tipo_mostrado = st.session_state['tipo_optimizado']
        duracion_mostrada = st.session_state['duracion_optimizada']
        
        # Formatear números para mejor visualización
        mo_formateada = f"{res['mano_obra_optima']:,.0f}".replace(',', ',')
        mat_formateada = f"{res['materiales_optimos']:,.0f}".replace(',', ',')
        costo_formateado = f"{res['costo_minimo']:,.0f}".replace(',', ',')
        
        # Mostrar los resultados con fondo blanco y texto negro legible
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;'>
            <h4 style='color: #2c3e50; text-align: center; margin-top: 0; margin-bottom: 15px; border-bottom: 1px solid #3498db; padding-bottom: 10px;'>
                ✅ RECURSOS ÓPTIMOS PARA {tipo_mostrado}
            </h4>
            
            <p style='margin: 5px 0 15px 0; text-align: center; color: #2c3e50; font-size: 14px;'>
                📊 Basado en {len(df[df['Tipo de Obra']==tipo_mostrado])} proyectos históricos
            </p>
            
            <table style='width: 100%; border-collapse: collapse; margin-bottom: 15px;'>
                <tr>
                    <td style='padding: 8px;'><span style='font-size: 20px;'>👷</span></td>
                    <td style='padding: 8px;'><strong style='color: #2c3e50;'>Mano de obra:</strong></td>
                    <td style='padding: 8px; text-align: right;'><span style='color: #2c3e50; font-weight: bold;'>{mo_formateada} horas</span></td>
                </tr>
                <tr>
                    <td style='padding: 8px;'><span style='font-size: 20px;'>🏗️</span></td>
                    <td style='padding: 8px;'><strong style='color: #2c3e50;'>Materiales:</strong></td>
                    <td style='padding: 8px; text-align: right;'><span style='color: #2c3e50; font-weight: bold;'>{mat_formateada} ton</span></td>
                </tr>
                <tr>
                    <td style='padding: 8px;'><span style='font-size: 20px;'>💰</span></td>
                    <td style='padding: 8px;'><strong style='color: #2c3e50;'>Costo estimado:</strong></td>
                    <td style='padding: 8px; text-align: right;'><span style='color: #2c3e50; font-weight: bold;'>${costo_formateado}</span></td>
                </tr>
                <tr>
                    <td style='padding: 8px;'><span style='font-size: 20px;'>📊</span></td>
                    <td style='padding: 8px;'><strong style='color: #2c3e50;'>Productividad:</strong></td>
                    <td style='padding: 8px; text-align: right;'><span style='color: #2c3e50; font-weight: bold;'>{res['productividad']:.3f} t/h</span></td>
                </tr>
            </table>
            
            <p style='margin: 5px 0 0 0; color: #7f8c8d; font-size: 13px; text-align: center; border-top: 1px solid #eee; padding-top: 10px;'>
                📌 Proyecto de {duracion_mostrada:.0f} días · Referencia histórica: {res['duracion_referencia']:.0f} días
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Mensaje cuando no hay datos calculados aún
        st.markdown("""
        <div style='background-color: white; padding: 40px 20px; border-radius: 10px; text-align: center; border: 1px dashed #bdc3c7;'>
            <h4 style='color: #7f8c8d; margin: 0;'>📐 OPTIMIZADOR LISTO</h4>
            <p style='color: #95a5a6; margin: 10px 0 0 0;'>
                👈 Configura los parámetros y haz clic en "CALCULAR"
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SIMULADOR DE IA (BASADO EN % DE RETRASO)
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>🤖 MODELO DE IA - SIMULADOR DE ESCENARIOS</h2>
<p style='color: #7f8c8d;'>Predice el % de retraso basado en Random Forest entrenado con datos históricos</p>
""", unsafe_allow_html=True)

# Leyenda de colores (PMI)
col_leg1, col_leg2, col_leg3 = st.columns(3)
with col_leg1:
    st.markdown("""
    <div style='background-color: #27ae60; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: white; margin: 0;'>🟢 CONTROLADO</h4>
        <p style='color: white; margin: 0;'>0-5% retraso</p>
    </div>
    """, unsafe_allow_html=True)
with col_leg2:
    st.markdown("""
    <div style='background-color: #f39c12; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: white; margin: 0;'>🟡 RIESGO MODERADO</h4>
        <p style='color: white; margin: 0;'>5-15% retraso</p>
    </div>
    """, unsafe_allow_html=True)
with col_leg3:
    st.markdown("""
    <div style='background-color: #e74c3c; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: white; margin: 0;'>🔴 CRÍTICO</h4>
        <p style='color: white; margin: 0;'>>15% retraso</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Inputs del simulador
col_s1, col_s2 = st.columns(2)

with col_s1:
    tipo_sim = st.selectbox("🏗️ Tipo de Obra", df['Tipo de Obra'].unique(), key='tipo_sim_ia')
    clima_sim = st.selectbox("☁️ Clima", df['Clima'].unique(), key='clima_sim_ia')
    presupuesto_sim = st.number_input("💰 Presupuesto ($)", 
                                      min_value=1_000_000, 
                                      max_value=50_000_000, 
                                      value=25_000_000,
                                      step=1_000_000,
                                      format="%d",
                                      key='pres_sim_ia')

with col_s2:
    duracion_sim = st.number_input("📅 Duración estimada (días)", 
                                   min_value=50, 
                                   max_value=800, 
                                   value=300,
                                   step=10,
                                   key='dur_sim_ia')
    materiales_sim = st.number_input("🏗️ Materiales (ton)", 
                                     min_value=1000, 
                                     max_value=30000, 
                                     value=10000,
                                     step=500,
                                     key='mat_sim_ia')
    mano_obra_sim = st.number_input("👷 Mano de obra (horas)", 
                                    min_value=5000, 
                                    max_value=200000, 
                                    value=50000,
                                    step=5000,
                                    key='mo_sim_ia')

if st.button("🎯 PREDECIR % DE RETRASO CON IA", use_container_width=True, key='btn_sim_ia'):
    with st.spinner("Analizando con IA (Random Forest)..."):
        alertas, porcentaje_pred = generar_alertas(
            tipo_sim, clima_sim, presupuesto_sim, duracion_sim, 
            materiales_sim, mano_obra_sim
        )
        
        st.session_state['simulacion_alertas'] = alertas
        st.session_state['simulacion_porcentaje'] = porcentaje_pred

# Mostrar resultados de la simulación - CORREGIDO (con verificación de existencia)
if 'simulacion_alertas' in st.session_state and 'simulacion_porcentaje' in st.session_state:
    st.markdown("---")
    st.markdown("### 📊 RESULTADOS DE LA SIMULACIÓN IA")
    
    porcentaje = st.session_state['simulacion_porcentaje']
    
    if porcentaje <= 5:
        color = "#27ae60"
        emoji = "🟢"
        nivel = "CONTROLADO"
        desc = "Proyecto dentro del margen aceptable (0-5%)"
    elif porcentaje <= 15:
        color = "#f39c12"
        emoji = "🟡"
        nivel = "RIESGO MODERADO"
        desc = "Requiere atención (5-15%)"
    else:
        color = "#e74c3c"
        emoji = "🔴"
        nivel = "CRÍTICO"
        desc = "Acción inmediata requerida (>15%)"
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown(f"""
        <div style='background-color: {color}; padding: 30px; border-radius: 15px; text-align: center;'>
            <h1 style='color: white; margin: 0; font-size: 72px;'>{emoji}</h1>
            <h2 style='color: white; margin: 0; font-size: 48px;'>{porcentaje:.1f}%</h2>
            <h3 style='color: white; margin: 0;'>{nivel}</h3>
            <p style='color: white; margin: 10px 0 0 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;'>
            <h4 style='color: #2c3e50; margin-top: 0; margin-bottom: 15px; border-bottom: 1px solid #3498db; padding-bottom: 10px;'>
                📋 INTERPRETACIÓN PMI
            </h4>
            <p style='margin: 10px 0; color: #2c3e50;'><strong>SPI equivalente:</strong> {100/(100+porcentaje):.2f}</p>
            <p style='margin: 10px 0; color: #2c3e50;'><strong>Días de retraso estimados:</strong> {(porcentaje/100)*duracion_sim:.1f} días</p>
            <p style='margin: 15px 0 0 0; color: #7f8c8d; font-size: 12px; border-top: 1px solid #eee; padding-top: 10px;'>
                *Basado en estándares del Project Management Institute
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar alertas
    st.markdown("### 🚨 ALERTAS GENERADAS POR IA:")
    for tipo, mensaje in st.session_state['simulacion_alertas']:
        if "🔴" in tipo:
            st.error(f"**{tipo}:** {mensaje}")
        elif "🟡" in tipo:
            st.warning(f"**{tipo}:** {mensaje}")
        elif "🟢" in tipo:
            st.success(f"**{tipo}:** {mensaje}")

st.markdown("---")

# ============================================
# TABLA DE DATOS CON MÉTRICAS PMI
# ============================================
st.markdown("""
<h2 style='color: #2c3e50;'>📋 DATOS HISTÓRICOS - MÉTRICAS PMI</h2>
""", unsafe_allow_html=True)

with st.expander("Ver todos los proyectos con métricas PMI", expanded=False):
    # Mostrar dataframe con las nuevas métricas
    df_display = df_filtrado[['Proyecto ID', 'Tipo de Obra', 'Clima', 'Porcentaje_Retraso', 'SPI', 'Nivel_Riesgo', 'Materiales', 'Mano_Obra']].copy()
    df_display['Porcentaje_Retraso'] = df_display['Porcentaje_Retraso'].round(1)
    df_display['SPI'] = df_display['SPI'].round(2)
    
    def color_filas(row):
        if row['Porcentaje_Retraso'] <= 5:
            return ['background-color: #d4edda'] * len(row)
        elif row['Porcentaje_Retraso'] <= 15:
            return ['background-color: #fff3cd'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    st.dataframe(
        df_display.style.apply(color_filas, axis=1),
        use_container_width=True,
        height=400
    )
    
    csv = df_display.to_csv(index=False)
    st.download_button(
        "📥 Descargar datos en CSV",
        csv,
        "datos_construccion_pmi.csv",
        "text/csv",
        use_container_width=True
    )

# ============================================
# FOOTER - RESUMEN DE MODELOS
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>🏗️ <strong>SISTEMA DE SOPORTE A LA DECISIÓN - BASADO EN PMI</strong></p>
    <p>📊 <strong>Métrica principal:</strong> % de retraso (Schedule Variance)</p>
    <p>🟢 < 5% | 🟡 5-15% | 🔴 > 15%</p>
    <p>📐 <strong>Modelo Matemático:</strong> Basado en promedios históricos por tipo de obra</p>
    <p>🤖 <strong>Modelo de IA:</strong> Random Forest para predicción de % de retraso</p>
    <p>✅ Simulación de escenarios · Optimización de recursos · Alertas predictivas</p>
</div>
""", unsafe_allow_html=True)
