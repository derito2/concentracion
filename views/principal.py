import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go






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


st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    mergeNov = pd.read_parquet("mergeNov.parquet") 
    df1 = pd.read_parquet("transaNov_merged1.parquet")
    df2 = pd.read_parquet("transaNov_merged2.parquet")

    # Unirlos como si fuera un solo archivo
    trans_f = pd.concat([df1, df2], ignore_index=True)
    #trans_f = pd.read_parquet("transaNov_merged.parquet")
    casos_f = pd.read_parquet("casosNov_merged.parquet")
    return mergeNov, trans_f, casos_f

mergeNov, trans_f, casos_f = load_data()


# ===========================================
#   CREAR FILTROS UNIFICADOS (MISMAS COLUMNAS)
# ===========================================

with st.sidebar:
    st.image("danu.png", use_container_width=True)

    st.markdown("""
        <h2 style='text-align:center; color:black; margin-top:-10px;'>
            Página Descriptiva
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


### VISUALIZACIONES

#mergeNov = pd.read_parquet("mergeNov.parquet") 



# ============================
# Layout: 2 columnas para las visualizaciones
# ============================
col1, col2 = st.columns(2)

# =======================================================
# 1) Gráfica de barras CHURN vs NO CHURN (con filtros)
# =======================================================
with col1:
    st.subheader("Churn vs No Churn")

    churn_count = (
        merge_f.groupby("UsuarioChurn").size().reset_index(name="count")
    )
    churn_count["UsuarioChurn"] = churn_count["UsuarioChurn"].map({0:"No Churn", 1:"Churn"})

    fig_churn = px.bar(
        churn_count,
        x="UsuarioChurn",
        y="count",
        #text="count",
        color="UsuarioChurn",
        color_discrete_sequence=["#31356E", "#070A4A"],
    )

    fig_churn.update_traces(
        textposition="outside",
        textfont=dict(size=14, family="Arial", color="#000000"),
        hovertemplate="<b>%{x}</b><br>Total: %{y}<extra></extra>",
        hoverlabel=dict(bgcolor="white", font_color="black")
    )

    fig_churn.update_layout(
        template="plotly_white",
        showlegend=False,
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
    )

    fig_churn.update_xaxes(
    tickfont=dict(size=14, color="black"),
    title=dict(text= "",font=dict(color="black")),
    gridcolor="rgba(0,0,0,0.08)",    # gris MUY suave como la gráfica izquierda
    zerolinecolor="rgba(0,0,0,0.15)"
    )   
    fig_churn.update_yaxes(
    tickfont=dict(size=14, color="black"),
    title=dict(font=dict(color="black")),
    gridcolor="rgba(0,0,0,0.08)",    # gris MUY suave como la gráfica izquierda
    zerolinecolor="rgba(0,0,0,0.15)"
    )


    st.plotly_chart(fig_churn, use_container_width=True)



# =======================================================
# 2) (Top 3 tipificaciones por mes)
# =======================================================

with col2:

    st.subheader("Top 3 Tipificaciones por Mes")

    casos_f["Fecha"] = pd.to_datetime(casos_f["Fecha"])
    casos_f["Mes"] = casos_f["Fecha"].dt.to_period("M").astype(str)

    counts = casos_f.groupby(["Mes", "tipificacion_proceso"]).size().reset_index(name="count")

    top3 = (
        counts.sort_values(["Mes", "count"], ascending=[True, False])
              .groupby("Mes")
              .head(3)
    )

    pivot = top3.pivot(index="Mes", columns="tipificacion_proceso", values="count").fillna(0)
    pivot = pivot.sort_index()

    color_scale = ["#070A4A", "#2D3170", "#48549E", "#6D75E0", "#A5B1EE", "#E6E7EE",
                   "#818288", "#4F4F50", "#080808"]

    fig_stacked = px.bar(
        pivot,
        x=pivot.index,
        y=pivot.columns,
        barmode="stack",
        color_discrete_sequence=color_scale,
    )

    fig_stacked.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        xaxis=dict(title="", tickfont=dict(size=14, color="black")),
        yaxis=dict(title="", tickfont=dict(size=14, color="black")),
        legend=dict(
            title=dict(text=""),
            font=dict(size=15, color="black"),
            bgcolor="white",
            bordercolor="white"
        )
    )

    fig_stacked.update_xaxes(color="black",gridcolor="rgba(0,0,0,0.08)",zerolinecolor="rgba(0,0,0,0.15)")
    fig_stacked.update_yaxes(color="black",gridcolor="rgba(0,0,0,0.08)",zerolinecolor="rgba(0,0,0,0.15)")

    st.plotly_chart(fig_stacked, use_container_width=True)

####
## TERCERA VISUALIZACION
# =======================================================
import plotly.graph_objects as go

st.markdown("---")
st.subheader("Usuarios Activos vs Monto Total — Análisis de Dinámica Mensual")

# ============================
# Cargar dataset
# ============================
#transa = pd.read_parquet("transaNov.parquet")

# Asegurar fecha correcta y derivar Mes
trans_f["fechaf"] = pd.to_datetime(trans_f["fechaf"])
trans_f["Mes"] = trans_f["fechaf"].dt.to_period("M").astype(str)

# ============================
# 1) Usuarios activos por mes
# ============================
usuarios_mes = (
    trans_f.groupby("Mes")["id_user"].nunique().reset_index(name="usuarios_activos")
)

# ============================
# 2) Total amount por mes
# ============================
amount_mes = (
    trans_f.groupby("Mes")["amount"].sum().reset_index(name="amount_total")
)

# Merge
df_line = usuarios_mes.merge(amount_mes, on="Mes")
df_line = df_line.sort_values("Mes").reset_index(drop=True)

# ============================
# 3) Cálculo de nuevos y perdidos mensuales
# ============================

users_por_mes = (
    trans_f.groupby("Mes")["id_user"]
    .apply(lambda x: set(x.unique()))
    .to_dict()
)

new_users_list = []
lost_users_list = []

meses = list(df_line["Mes"])

for i in range(len(meses)):
    if i == 0:
        new_users_list.append(0)
        lost_users_list.append(0)
    else:
        mes_actual = users_por_mes[meses[i]]
        mes_anterior = users_por_mes[meses[i - 1]]

        new_users = mes_actual - mes_anterior
        lost_users = mes_anterior - mes_actual

        new_users_list.append(len(new_users))
        lost_users_list.append(len(lost_users))

df_line["usuarios_nuevos"] = new_users_list
df_line["usuarios_perdidos"] = lost_users_list


# ============================
# 4) Plotly: barras + línea con tooltip personalizado
# ============================
base_color = "#070A4A"
light_color = "#6D75E0"

fig = go.Figure()

# === Barras: usuarios activos ===
fig.add_trace(
    go.Bar(
        x=df_line["Mes"],
        y=df_line["usuarios_activos"],
        name="Usuarios Activos",
        marker=dict(color=base_color, opacity=1),
        hovertemplate=
        "<b>Mes:</b> %{x}<br>" +
        "<b>Usuarios Activos:</b> %{y}<br>" +
        "<b>Nuevos:</b> %{customdata[0]}<br>" +
        "<b>Perdidos:</b> %{customdata[1]}<br>" +
        "<extra></extra>",
        customdata=df_line[["usuarios_nuevos", "usuarios_perdidos"]].values
    )
)

# === Línea: Amount ===
fig.add_trace(
    go.Scatter(
        x=df_line["Mes"],
        y=df_line["amount_total"],
        name="Monto Total",
        yaxis="y2",
        mode="lines+markers",
        line=dict(width=3, color=light_color),
        marker=dict(size=8),
        hovertemplate="<b>Mes:</b> %{x}<br><b>Monto:</b> %{y}<extra></extra>"
    )
)

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),

    xaxis=dict(
        title="",
        tickfont=dict(size=16, color="black"),
        color="black",
        title_font=dict(size=16, color="black")
    ),
    yaxis=dict(
        title="Usuarios Activos",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=16, color="black"),
        color="black",
        domain=[0, 1],
        gridcolor="rgba(0,0,0,0.08)",    # gris MUY suave como la gráfica izquierda
        zerolinecolor="rgba(0,0,0,0.15)"
    ),
    yaxis2=dict(
        title="Monto Total",
        overlaying="y",
        side="right",
        #anchor="x",
        showgrid=False,
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=16, color="black"),
        color="black",
        domain=[0, 1],
        gridcolor="rgba(0,0,0,0.08)",    # gris MUY suave como la gráfica izquierda
        zerolinecolor="rgba(0,0,0,0.15)"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=1,
        font=dict(size=15, color="black")
    )
)

st.plotly_chart(fig, use_container_width=True)
