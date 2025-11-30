import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns




st.set_page_config(layout="wide")
import plotly.io as pio
pio.templates.default = "simple_white"


light_css = """
<style>

/* ===========================================
   Fondo general — AHORA EN BLANCO
   =========================================== */
[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
    color: #1A1A1A;
}

[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0.9);
    color: #1A1A1A;
}



/* Quita padding global del contenedor principal */
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* Quita el margen superior que Streamlit mete entre componentes */
    div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* Quitar padding extra entre columnas */
    div[data-testid="column"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* Quita margen superior al bloque donde están los KPIs */
    div[data-testid="stMetric"] {
        margin-top: 4rem !important;
    }

    



    /* Fondo del sidebar */
section[data-testid="stSidebar"] {
    background-color: #E6E7EE !important;
}

/* Títulos del sidebar */
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h1 {
    color: #48549E !important;
}

/* Textos y labels */
section[data-testid="stSidebar"] label {
    color: #48549E!important;
    font-weight: 600;
}

/* Inputs del sidebar */
section[data-testid="stSidebar"] .stMultiSelect,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] input {
    color: #48549E !important;
}


/* Cambiar color de la barra del slider */
div[data-baseweb="slider"] > div > div {
    background-color: #6D75E0 !important;      /* color de la barra */
}

/* Cambiar color de la barra seleccionada */
div[data-baseweb="slider"] > div > div > div {
    background-color: #A5B1EE !important;      /* color del segmento activo */
}

/* Cambiar color del thumb (la bolita) */
div[data-baseweb="slider"] [role="slider"] {
    background-color: #6D75E0 !important;
    border: 2px solid #A5B1EE !important;
}

/* Cambiar color del texto del valor */
div[data-baseweb="slider"] span {
    color: black !important;
}

/* ===== Tooltip del valor cuando mueves el slider ===== */
div[data-baseweb="slider"] div[role="slider"] > div {
    background-color: transparent !important;      /* fondo del tooltip */
    color: #6D75E0 !important;                 /* color del número */
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}


/* Caja del select/multiselect (fondo + borde) */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #F3F4FD !important;    /* Fondo */
    border: 1.5px solid #6D75E0 !important;  /* Borde */
    border-radius: 6px !important;
}

/* Texto dentro de la caja */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #6D75E0 !important;
}

/* Fondo del menú desplegable */
section[data-testid="stSidebar"] [role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #6D75E0 !important;
}

/* Texto de las opciones del menú */
section[data-testid="stSidebar"] [role="listbox"] * {
    color: #6D75E0 !important;
}

/* Chips de multiselect */
section[data-testid="stSidebar"] div[data-baseweb="tag"] {
    background-color: #A5B1EE !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}


</style>
"""
st.markdown(light_css, unsafe_allow_html=True)

# -------------------------
# CARGA DE DATOS (cacheada)
# -------------------------
@st.cache_data
def load_data():
    mergeNov = pd.read_parquet("mergeNovClient.parquet")
    df1 = pd.read_parquet("transaNov_merged1.parquet")
    df2 = pd.read_parquet("transaNov_merged2.parquet")

    # Unirlos como si fuera un solo archivo
    trans_f = pd.concat([df1, df2], ignore_index=True)
    #trans_f = pd.read_parquet("transaNov_merged.parquet")
    users = pd.read_parquet("usersNov.parquet")
    casos_f = pd.read_parquet("casosNov_merged.parquet")
    return mergeNov, trans_f, users, casos_f

mergeNov, trans_f, users, casos_f = load_data()


# ===========================================
#   CREAR FILTROS UNIFICADOS (MISMAS COLUMNAS)
# ===========================================

with st.sidebar:
    st.image("danu.png", use_container_width=True)

    st.markdown("""
        <h2 style='text-align:center; color:black; margin-top:-10px;'>
            Página Clientes
        </h2>
        <hr style="margin-top:2px;">
    """, unsafe_allow_html=True)



with st.sidebar:
    st.header("Filtros globales")

    gender_list = trans_f["gender"].dropna().unique().tolist()
    qualification_list = trans_f["qualification"].dropna().unique().tolist()
    usertype_list = trans_f["usertype"].dropna().unique().tolist()
    state_list = trans_f["state"].dropna().unique().tolist()

    f_gender = st.multiselect("Género", gender_list)
    f_qualification = st.multiselect("Nivel de cuenta", qualification_list)
    f_usertype = st.multiselect("Tipo de usuario", usertype_list)
    f_state = st.multiselect("Estado", state_list)

    # filtro de edad (usar min y max reales)
    min_age = int(trans_f["age"].min())
    max_age = int(trans_f["age"].max())
    f_age = st.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

# ============================================
# FUNCIÓN PARA APLICAR FILTROS A CUALQUIER DF
# ============================================
def aplicar_filtros(df):
    if "gender" in df.columns and f_gender:
        df = df[df["gender"].isin(f_gender)]
    if "qualification" in df.columns and f_qualification:
        df = df[df["qualification"].isin(f_qualification)]
    if "usertype" in df.columns and f_usertype:
        df = df[df["usertype"].isin(f_usertype)]
    if "state" in df.columns and f_state:
        df = df[df["state"].isin(f_state)]
    if "age" in df.columns:
        df = df[(df["age"] >= f_age[0]) & (df["age"] <= f_age[1])]
    return df

# ============================
# Aplicar filtros a cada df
# ============================
trans_f = aplicar_filtros(trans_f.copy())
casos_f = aplicar_filtros(casos_f.copy())
merge_f = aplicar_filtros(mergeNov.copy())



# ================================
#   KPI – Usuarios activos últimos 42 vs anteriores 42
# ================================

# Cargar dataset
#trans = pd.read_parquet("transaNov_merged.parquet")

# Asegurar que la fecha es datetime
trans_f["fechaf"] = pd.to_datetime(trans_f["fechaf"])

# Ordenar por fecha descendente
trans_f = trans_f.sort_values("fechaf", ascending=False)

# Fecha más reciente
max_date = trans_f["fechaf"].max()

# Definir ventanas
window_42_days = pd.Timedelta(days=42)
window_84_days = pd.Timedelta(days=84)

# Filtro 84 días recientes
mask_84 = trans_f["fechaf"] >= (max_date - window_84_days)
trans_84 = trans_f[mask_84]

# Dividir en dos ventanas
mask_last_42 = trans_84["fechaf"] >= (max_date - window_42_days)
mask_prev_42 = (trans_84["fechaf"] < (max_date - window_42_days)) & \
               (trans_84["fechaf"] >= (max_date - window_84_days))

trans_last_42 = trans_84[mask_last_42]
trans_prev_42 = trans_84[mask_prev_42]

# Usuarios únicos por ventana
users_last_42 = trans_last_42["id_user"].nunique()
users_prev_42 = trans_prev_42["id_user"].nunique()

# Cálculo del cambio porcentual
if users_prev_42 == 0:
    pct_change = 0
else:
    pct_change = ((users_last_42 - users_prev_42) / users_prev_42) * 100


# ==========================================
#   KPI 2 y KPI 3 – Cálculo por usuario
# ==========================================

# Asegurar datetime
trans_f["fechaf"] = pd.to_datetime(trans_f["fechaf"])

# Obtener año-mes más reciente
trans_f["year_month"] = trans_f["fechaf"].dt.to_period("M")
latest_month = trans_f["year_month"].max()
previous_month = latest_month - 1

# Filtrar meses
df_latest = trans_f[trans_f["year_month"] == latest_month]
df_previous = trans_f[trans_f["year_month"] == previous_month]

# ======================================================
#   AGRUPACIÓN POR USUARIO
# ======================================================

# ---- MONTH LATEST ----
latest_group = df_latest.groupby("id_user").agg({
    "amount": "sum",
    "trnx": "sum"
}).reset_index()

# ---- MONTH PREVIOUS ----
prev_group = df_previous.groupby("id_user").agg({
    "amount": "sum",
    "trnx": "sum"
}).reset_index()

# ======================================================
#   KPI 2 – Mediana del amount por usuario
# ======================================================

median_latest = latest_group["amount"].median() if not latest_group.empty else 0
median_previous = prev_group["amount"].median() if not prev_group.empty else 0

# Cambio porcentual
if median_previous == 0:
    pct_change_median = 0
else:
    pct_change_median = ((median_latest - median_previous) / median_previous) * 100


# ======================================================
#   KPI 3 – Promedio de transacciones por usuario
# ======================================================

avg_latest = latest_group["trnx"].mean() if not latest_group.empty else 0
avg_previous = prev_group["trnx"].mean() if not prev_group.empty else 0

if avg_previous == 0:
    pct_change_avg = 0
else:
    pct_change_avg = ((avg_latest - avg_previous) / avg_previous) * 100


# =====================================================
#   KPI 4 y KPI 5 — usando casosNov.parquet
# =====================================================

import pandas as pd
import streamlit as st

# Cargar archivo
#casos = pd.read_parquet("casosNov_merged.parquet")

# Asegurar datetime
casos_f["Fecha"] = pd.to_datetime(casos_f["Fecha"])

# Crear columna año-mes
casos_f["year_month"] = casos_f["Fecha"].dt.to_period("M")

# Detectar mes más reciente
latest_month_cases = casos_f["year_month"].max()
previous_month_cases = latest_month_cases - 1

# Filtrar meses
df_latest_cases = casos_f[casos_f["year_month"] == latest_month_cases]
df_previous_cases = casos_f[casos_f["year_month"] == previous_month_cases]

# ==========================================
#   KPI 4 – Conteo de casos por mes
# ==========================================

count_cases_latest = len(df_latest_cases)
count_cases_previous = len(df_previous_cases)

# Cambio porcentual
if count_cases_previous == 0:
    pct_change_cases = 0
else:
    pct_change_cases = ((count_cases_latest - count_cases_previous) / count_cases_previous) * 100





# ================================
#   Mostrar KPI con estilo
# ================================

st.markdown("""
<style>

    /* === Caja principal del KPI === */
    div[data-testid="stMetric"] {
        height: 110px !important;
        padding: 5px !important;

        display: flex !important;
        flex-direction: column !important;
        justify-content: space-evenly !important;
        align-items: center !important;

        border: 4px solid #31356E;
        border-radius: 12px;
        background-color: #070A4A;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);

        text-align: center !important;
        overflow: hidden !important;
        width: 100%;
    }

    /* === Aplica el mismo INTERLINEADO a todo texto dentro del KPI === */
    div[data-testid="stMetric"] * {
        line-height: 0.90 !important;
    }

    /* === LABEL === */
    div[data-testid="stMetric"] > label > p {
        font-size: 18px !important;
        margin: 0 0 2px 0 !important;
        padding: 0 !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        color: white !important;
    }

    /* === VALUE === */
    div[data-testid="stMetric"] > div:first-of-type {
        font-size: 28px !important;
        margin: 0 0 2px 0 !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        color: white !important;
    }

    /* === DELTA === */
    div[data-testid="stMetric"] > div:nth-of-type(2) {
        font-size: 16px !important;
        margin: 0 !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        text-align: center !important;
        width: 100%;
    }

</style>
""", unsafe_allow_html=True)


col1, col2, col3, col4 = st.columns(4)



with col1:
    st.metric(
        label="Usuarios Activos últimos 42 días",
        value=f"{users_last_42:,}",
        delta=f"{pct_change:.2f}%",
        delta_color="normal"
    )

with col2:
    st.metric(
        label=f"Mediana Amount por Usuario – {latest_month}",
        value=f"${median_latest:,.2f}",
        delta=f"{pct_change_median:.2f}%",
        delta_color="normal"
    )


with col3:
    st.metric(
        label=f"Promedio Transacciones por Usuario – {latest_month}",
        value=f"{avg_latest:,.2f}",
        delta=f"{pct_change_avg:.2f}%",
        delta_color="normal"
    )

with col4:
    st.metric(
        label=f"Casos Contact Center – {latest_month_cases}",
        value=f"{count_cases_latest:,}",
        delta=f"{pct_change_cases:.2f}%",
        delta_color="inverse"
    )

# ============================================================
# 2. LIMPIEZA / VARIABLES DERIVADAS (USANDO merge_f)
# ============================================================

merge = merge_f.copy()  # usamos la versión ya filtrada por tus filtros globales

# llamadas al año
merge["calls_total_año"] = merge["calls_total"] * 12

# UsuarioChurn a numérico
merge["UsuarioChurn"] = pd.to_numeric(merge["UsuarioChurn"], errors="coerce")

# Edad
merge["age"] = pd.to_numeric(merge.get("age"), errors="coerce")

# Género estándar
if "gender" in merge.columns:
    merge["gender"] = (
        merge["gender"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan})
    )

# user_type: alias si viene como "usertype"
if "user_type" not in merge.columns and "usertype" in merge.columns:
    merge["user_type"] = merge["usertype"]

# qualification a string limpia
if "qualification" in merge.columns:
    merge["qualification"] = (
        merge["qualification"]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan})
    )

# Lista de columnas de montos por tipo
cols_montos = [
    "Retiro_amt",
    "Envio de Dinero_amt",
    "Deposito_amt",
    "Compra in App_amt",
    "Compra con Tarjeta_amt"
]

# ============================================================
# 3. DICCIONARIO DE ESTADOS
# ============================================================

mapa_estados = {
    'AG':'Aguascalientes','BC':'Baja California','BS':'Baja California Sur','CM':'Campeche',
    'CO':'Coahuila','CL':'Colima','CS':'Chiapas','CH':'Chihuahua','DF':'Ciudad de México',
    'DG':'Durango','GT':'Guanajuato','GR':'Guerrero','HG':'Hidalgo','JC':'Jalisco','MX':'México',
    'MI':'Michoacán','MO':'Morelos','NA':'Nayarit','NL':'Nuevo León','OA':'Oaxaca','PU':'Puebla',
    'QE':'Querétaro','QR':'Quintana Roo','SL':'San Luis Potosí','SI':'Sinaloa','SO':'Sonora',
    'TB':'Tabasco','TM':'Tamaulipas','TL':'Tlaxcala','VE':'Veracruz','YU':'Yucatán','ZA':'Zacatecas'
}

merge["state_full"] = merge["state"].map(mapa_estados)

# ============================================================
# 5. NO APLICAMOS FILTROS AQUÍ (YA ESTÁN APLICADOS)
# ============================================================

merge_filt = merge.copy()

if merge_filt.empty:
    st.warning("No hay usuarios que cumplan los filtros; se muestran todos los registros.")
    merge_filt = merge.copy()


st.markdown("---")

# ============================================================
# 6. MÉTRICAS POR ESTADO (USANDO merge_filt)
# ============================================================

monto_estado = (
    merge_filt
    .groupby("state")["amount_total"]
    .mean()
    .reset_index()
    .rename(columns={"amount_total": "Monto promedio al mes"})
)

llamadas_estado = (
    merge_filt
    .groupby("state")["calls_total_año"]
    .sum()
    .reset_index()
)

llamadas_estado["Llamadas promedio al mes"] = (
    llamadas_estado["calls_total_año"] / 12
)
llamadas_estado = llamadas_estado[["state", "Llamadas promedio al mes"]]

metricas_estado = monto_estado.merge(llamadas_estado, on="state", how="left")

# Churn por estado
churn_rate_estado = (
    merge_filt
    .groupby("state")["UsuarioChurn"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={
        "mean": "Porcentaje CHURN (%)",
        "count": "Total de Usarios CHURN"
    })
)

churn_rate_estado["Porcentaje CHURN (%)"] *= 100
churn_rate_estado["state_full"] = churn_rate_estado["state"].map(mapa_estados)
metricas_estado["state_full"]   = metricas_estado["state"].map(mapa_estados)

df_final = churn_rate_estado.merge(
    metricas_estado,
    on=["state", "state_full"],
    how="left"
)

df_final = df_final[df_final["state_full"].notna()].copy()
df_final["state_full"] = df_final["state_full"].astype(str)

# ============================================================
# 7. FUNCIÓN PARA DETECTAR CAMPO DEL GEOJSON
# ============================================================

def detectar_campo_geojson(mx_geo, df_estados, col_estados="state_full"):
    estados = set(df_estados[col_estados].dropna().astype(str).unique())
    if not estados:
        return "name"

    ejemplo_props = mx_geo["features"][0]["properties"]
    coincidencias = []
    for key in ejemplo_props.keys():
        valores = {
            str(feat["properties"].get(key, "")).strip()
            for feat in mx_geo["features"]
        }
        inter = estados.intersection(valores)
        coincidencias.append((key, len(inter)))

    coincidencias.sort(key=lambda x: x[1], reverse=True)
    mejor_key, n_match = coincidencias[0]
    return mejor_key if n_match > 0 else "name"

# ============================================================
# 8. MAPA DE CHURN POR ESTADO (FILTRADO)
# ============================================================

#st.subheader("Mapa de churn por estado (usuarios filtrados)")

if df_final.empty:
    st.info("No hay datos para mostrar el mapa con los filtros seleccionados.")
else:
    with open("mx.json", "r", encoding="utf-8") as f:
        mx_geo = json.load(f)

    campo_geo = detectar_campo_geojson(mx_geo, df_final, "state_full")
    #st.caption(f"Unión con GeoJSON usando la propiedad: '{campo_geo}'")

    # Paleta degradada personalizada (tus colores)
    custom_colors = ["#8FA6FF","#48549E","#070A4A"]

    fig_mapa = px.choropleth(
        df_final,
        geojson=mx_geo,
        featureidkey=f"properties.{campo_geo}",
        locations="state_full",
        color="Porcentaje CHURN (%)",
        hover_name="state_full",
        hover_data={
            "Porcentaje CHURN (%)": ':.2f',
            "Total de Usarios CHURN": True,
            "Monto promedio al mes": ':.2f',
            "Llamadas promedio al mes": ':.2f',
            "state_full": False
        },
        color_continuous_scale=custom_colors,
        height=320
    )

    # Sin líneas de frontera y autozoom
    fig_mapa.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
    )

    # Tema oscuro + fondos transparentes
    fig_mapa.update_layout(
        template="simple_white",
        margin=dict(l=0, r=0, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',

        # === COLORBAR EN NEGRO ===
        coloraxis_colorbar=dict(
        title=dict(font=dict(color="black")),
        tickfont=dict(color="black")
        )
    )

    #st.plotly_chart(fig_mapa, use_container_width=True)

#st.markdown("---")



# ============================================================
# 🔹 1) PROCESAMIENTO (FUERA DE col1)
# ============================================================

if merge_filt.empty:
    df_violin = pd.DataFrame()  # Para evitar errores
    fig_vio = None
else:
    df_violin = merge_filt.copy()

    # Total manual
    df_violin["amount_total_manual"] = df_violin[cols_montos].sum(axis=1)

    # P95 y truncado
    p95 = df_violin["amount_total_manual"].quantile(0.95)
    df_violin["amount_total_trunc"] = df_violin["amount_total_manual"].clip(upper=p95)

    # Aportador mayor
    suma_por_tipo = df_violin[cols_montos].sum()
    mayor_aportador_col = suma_por_tipo.idxmax()
    porcentaje_mayor = (suma_por_tipo.max() / suma_por_tipo.sum()) * 100
    label_mayor = mayor_aportador_col.replace("_amt", "")

    # Categoría única para el violin
    df_violin["categoria"] = "Monto total (P95)"

    # ========= FIGURA VIOLIN PLOT ==========

# Estética
fig_violin = px.violin(
    df_violin,
    x="categoria",
    y="amount_total_trunc",
    color_discrete_sequence=["#48549E"],  # azul
    box=False,
    points=False,
)

fig_violin.update_layout(
    template="simple_white",

    # ====== FONDO ======
    paper_bgcolor="white",
    plot_bgcolor="white",

    # ====== FUENTE GENERAL ======
    font=dict(color="black", size=18),

    height=345,
    margin=dict(l=10, r=10, t=40, b=10),

    # ====== EJES (forzado a negro) ======
    xaxis=dict(
        title=dict(text="", font=dict(color="black")),
        tickfont=dict(size = 14,color="black"),
        linecolor="black",
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(text="Cantidad", font=dict(color="black")),
        tickfont=dict(size = 14,color="black"),
        linecolor="black",
        gridcolor="lightgray",
        zeroline=False
    ),
)

# ====== TRAZAS DEL VIOLÍN ======
fig_violin.update_traces(
    meanline_visible=True,
    line_color="black",     # bordes del violín -> negro
    width=0.25,
    fillcolor="rgba(72, 84, 158, 0.6)",
)




### REVENUE
# ========= PREPARAR FECHA PARA GRAFICAR ==========
df_bar = pd.read_parquet("df_bar_revenue.parquet")
df_bar.index = pd.PeriodIndex(df_bar.index, freq="M").to_timestamp()
df_bar["mes_fmt"] = df_bar.index.strftime("%B %Y")

# ========= CONVERTIR A FORMATO LONG PARA BARRAS HORIZONTALES ==========
df_bar_long = df_bar.melt(
    id_vars="mes_fmt",
    value_vars=["Revenue por Depósitos", "Revenue por Retiros"],
    var_name="tipo",
    value_name="valor"
)

# ========= FIGURA BARRAS STACKED HORIZONTALES ==========
fig_bar = px.bar(
    df_bar_long,
    x="valor",
    y="mes_fmt",
    color="tipo",
    orientation="h",
    color_discrete_sequence=["#070A4A", "#48549E"]
)

# ========= ESTÉTICA ==========
fig_bar.update_layout(
    barmode="stack",
    template="simple_white",

    # ======= FONDO FORZADO A BLANCO ========
    paper_bgcolor="white",
    plot_bgcolor="white",

    # ======= FONTS GENERALES ========
    font=dict(color="black", size=13),

    # ======= LEYENDA ========
    legend=dict(
        title=dict(text="", font=dict(color="black", size=15)),
        font=dict(color="black", size=15),
        orientation="h",
        yanchor="bottom",
        y=1.02
    ),

    # ======= TAMAÑO GLOBAL ========
    height=760,
    margin=dict(l=10, r=10, t=40, b=10),
)

# ======= FORMATO DEL EJE X ========
fig_bar.update_xaxes(
    title=dict(text="MXN", font=dict(color="black", size=16)),
    tickfont=dict(color="black", size=14)
)

# ======= FORMATO DEL EJE Y ========
fig_bar.update_yaxes(
    title=dict(text="Mes", font=dict(color="black", size=16)),
    tickfont=dict(color="black", size=14)
)


# ============================================================
# 🔹 2) VISUALIZACIÓN (DENTRO DE col1) — FORMATO TIPO EJEMPLO
# ============================================================


# ============================
# NUEVO LAYOUT
# ============================

col_left, col_right = st.columns([0.50, 0.50])  # 35% / 65%

# ======================================
# 📌 COLUMNA IZQUIERDA: SOLO REVENUE
# ======================================
with col_left:
    st.subheader("Revenue mensual (global)")
    st.plotly_chart(fig_bar, use_container_width=True)


# ======================================
# 📌 COLUMNA DERECHA: MAPA (ARRIBA) + VIOLIN (ABAJO)
# ======================================
with col_right:

    # ---- Mapa arriba ----
    st.subheader("Mapa de churn por estado (usuarios filtrados)")
    st.plotly_chart(fig_mapa, use_container_width=True)

    st.markdown("---")

    # ---- Violin abajo ----
    st.subheader("Distribución del monto total (P95)")
    st.plotly_chart(fig_violin, use_container_width=True)


#"#0000D1"
#"#767171"
