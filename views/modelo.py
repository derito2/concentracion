# ============================================
#       APP COMPLETA DE PREDICCIÓN DE CHURN
#               (SIN FILTROS)
# ============================================

import streamlit as st
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import re
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import os




# ============================================
# CONFIGURACIÓN DE LA APP
# ============================================

st.set_page_config(layout="wide", page_title="Predicción de Churn - MLP")

light_css = """
<style>

/* ===========================================
   Fondo general
   =========================================== */
[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
    color: #1A1A1A;
}

[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0.9);
    color: #1A1A1A;
}

main h1, main h2, main h3, main h4, main h5, main h6,
main label, main p, main div {
    color: #1A1A1A !important;
}


h1 { font-size: 1.9rem; font-weight: 700; }
h2 { font-size: 1.4rem; }
h3 { font-size: 1.2rem; }

/* ===========================================
   File uploader
   =========================================== */
[data-testid="stFileUploader"] section {
    background-color: #F5F5F5;
    border-radius: 6px;
    border: 1px dashed #48549E;
}
[data-testid="stFileUploader"] * {
    color: #1A1A1A !important;
}

/* ===========================================
   Alertas
   =========================================== */
.stAlert {
    background-color: #F0F2F6 !important;
    color: #1A1A1A !important;
    border-left: 4px solid #48549E !important;
}

/* ===========================================
   Botones (st.button)
   =========================================== */
div.stButton > button,
div.stButton > button[kind="primary"],
div.stButton > button:active,
div.stButton > button:focus-visible {
    background-color:  #A5B1EE !important; 
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

div.stButton > button:hover {
    background-color: #B7A7C6 !important;
    color: white !important;
}

div.stButton > button:active {
    background-color: #B7A7C6 !important;
    color: white !important;
}

/* ===========================================
   BOTÓN DEL FILE UPLOADER (“Browse files”)
   =========================================== */
[data-testid="stFileUploader"] button {
    background-color: #070A4A !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #0B0F6A !important;
    color: white !important;
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



with st.sidebar:
    st.image("danu.png", use_container_width=True)

    st.markdown("""
        <h2 style='text-align:center; color:black; margin-top:-10px;'>
            Página Modelo ML
        </h2>
        <hr style="margin-top:2px;">
    """, unsafe_allow_html=True)

# ============================================
# DICCIONARIO DE EXPLICACIONES (no se usa aún)
# ============================================

explicaciones = {
    'Compra con Tarjeta_amt': 'Gasto presencial bajo',
    'Compra in App_trnx': 'Pocas compras en app',
    'Envio de Dinero_amt': 'Envía poco dinero',
    'Retiro_trnx': 'Pocos retiros',
    'amount_per_call': 'Depósitos/llamadas muy bajos',
    'cv_dias_entre_tx': 'Variabilidad extrema de uso',
    'iqr_dias_entre_tx': 'Inconsistencia en actividad',
    'mediana_dias_entre_tx': 'Intervalos usuales largos',
    'min_dias_entre_tx': 'Demasiado tiempo entre usos',
    'moda_dias_entre_tx': 'Frecuencia más común baja',
    'num_intervalos': 'Pocas transacciones totales',
    'p25_dias_entre_tx': 'Primeros días inactivos altos',
    'std_dias_entre_tx': 'Actividad muy irregular',
    'UsuarioChurn': 'Indicador de churn previo',

    'canalcontacto_mas_frecuente_LLamada': 'Llama mucho al soporte',

    'qualification_2': 'Nivel de cuenta 2',
    'qualification_3': 'Nivel de cuenta 3',

    'tipificacion_mas_frecuente_01 Consulta De Saldos Y Movimientos':
        'Llama por dudas de saldo',
    'tipificacion_mas_frecuente_07 Cancelacion De Tarjeta':
        'Interés en cancelar',
    'tipificacion_mas_frecuente_09 Consulta Sobre Informacion De Cuentas':
        'Quiere saber información',
    'tipificacion_mas_frecuente_14 Cliente En Lista Negra Incorporacion Notificacion De Rechazo':
        'Cliente en lista negra',
    'tipificacion_mas_frecuente_17 Actualizacion De Informacion De Cuenta':
        'Actualiza datos seguido',
    'tipificacion_mas_frecuente_20 Bloquear Y Desbloquear Cuenta':
        'Bloqueos frecuentes',
    'tipificacion_mas_frecuente_34 Desbloqueo De Pin Digital':
        'Problemas con PIN',
    'tipificacion_mas_frecuente_37 Pin Y Chip':
        'Problemas con chip',
    'tipificacion_mas_frecuente_38 Consulta De Saldo':
        'Consulta saldo seguido',
    'tipificacion_mas_frecuente_39 Acuatualizacion De Nip':
        'Cambio frecuente de NIP',
    'tipificacion_mas_frecuente_40 Actualizacion Numero Telefonico':
        'Cambios de teléfono',
    'tipificacion_mas_frecuente_Desconocido':
        'Motivo desconocido',
    'tipificacion_mas_frecuente_otros':
        'Motivos variados'
}


# ============================================
#             STREAMLIT APP
# ============================================

st.title("Predicción de Churn - MLP")
st.write("Sube tu archivo de datos nuevos y genera el riesgo de churn por usuario.")


# ============================================================
# 0. Intentar cargar el archivo predicciones_churn.parquet
# ============================================================

def cargar_df_resultado():
    """Carga df_resultado desde parquet si existe; si no, devuelve None."""
    try:
        if os.path.exists("predicciones_churn.parquet"):
            return pd.read_parquet("predicciones_churn.parquet")
        return None
    except Exception:
        return None


# Cargar df_resultado desde sesión o desde disco
if "df_resultado" not in st.session_state:
    st.session_state["df_resultado"] = cargar_df_resultado()


# ============================================================
# CARGA DE ARTIFACTOS DEL MODELO
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_mlp_model.pkl")
    column_order = joblib.load("column_order.pkl")
    onehot_cols = joblib.load("onehot_columns.pkl")
    return model, column_order, onehot_cols


model, column_order, onehot_cols = load_artifacts()


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
col_left, col_right = st.columns([1,2])


# ============================================================
# 1. LADO IZQUIERDO – CARGA ARCHIVO + EJECUTAR MODELO
# ============================================================
with col_left:
    st.subheader("Cargar archivo")

    uploaded_file = st.file_uploader("Subir archivo .parquet", type=["parquet"])
    df_original = None

    if uploaded_file is not None:
        df_original = pd.read_parquet(uploaded_file, engine="fastparquet")
        df_original = df_original.set_index("id_user", drop=True)
        st.metric("Usuarios cargados", f"{df_original.shape[0]:,}")

    correr = st.button("Ejecutar modelo")

    # ============================================================
    # SI EL USUARIO PRESIONA "Ejecutar modelo"
    # ============================================================
    if (df_original is not None) and correr:

        df_modelo_input = df_original.copy()
        df_proc = df_modelo_input.copy()

        # --- Limpieza antes de one-hot ---
        df_proc = df_proc.drop(
            ['agente_motivo_top', 'country', 'veces_churn'],
            axis=1,
            errors='ignore'
        )

        if "qualification" in df_proc.columns:
            df_proc["qualification"] = df_proc["qualification"].astype("category")

        if "UsuarioChurn" in df_proc.columns:
            df_proc["UsuarioChurn"] = df_proc["UsuarioChurn"].astype(bool)

        if "tipificacion_max_racha" in df_proc.columns:
            df_proc["tipificacion_max_racha"] = df_proc["tipificacion_max_racha"].astype("string")

        if (
            "max_llamadas_seguidas_misma_tipificacion" in df_proc.columns
            and "tipificacion_max_racha" in df_proc.columns
        ):
            mask = df_proc['max_llamadas_seguidas_misma_tipificacion'] == 1.0
            df_proc.loc[mask, 'tipificacion_max_racha'] = 'ninguno'

        if "tipificacion_max_racha" in df_proc.columns:
            df_proc["tipificacion_max_racha"] = df_proc["tipificacion_max_racha"].astype("category")

        df_proc = df_proc.drop(
            ["tipificacion_max_racha",
            "max_llamadas_seguidas_misma_tipificacion",
            "max_dias_entre_tx"],
            axis=1,
            errors='ignore'
        )

        # --- One-hot ---
        cat_cols = [
            c for c in df_proc.columns
            if df_proc[c].dtype.name in ["category", "object", "string"]
        ]

        df_proc = pd.get_dummies(
            df_proc,
            columns=cat_cols,
            drop_first=True,
            dtype=int
        )

        for col in onehot_cols:
            if col not in df_proc.columns:
                df_proc[col] = 0

        df_proc = df_proc[[c for c in onehot_cols if c != "UsuarioChurn"]]
        df_proc = df_proc[column_order]

        # ============================
        #      PREDICCIONES
        # ============================
        pred = model.predict(df_proc)
        proba = model.predict_proba(df_proc)[:, 1]

        df_pred = pd.DataFrame({
            "id_user": df_proc.index,
            "prediccion_churn": pred,
            "probabilidad_churn": proba,
        }).set_index("id_user")

        df_resultado_modelo = df_original.merge(
            df_pred, 
            left_index=True, 
            right_index=True, 
            how="left"
        )

        #############
        df_churn = df_proc[pred == 1]

        if df_churn.shape[0] > 0:
            usuario_muestra = df_churn.iloc[0].values

            explainer = LimeTabularExplainer(
                training_data=df_proc.values,
                feature_names=df_proc.columns,
                class_names=["No churn", "Churn"],
                mode="classification"
            )

            exp = explainer.explain_instance(
                usuario_muestra,
                model.predict_proba,
                num_features=30
            )

            lime_df = pd.DataFrame(exp.as_list(), columns=["variable", "peso"])
            lime_df = lime_df[lime_df["peso"] > 0]

            def limpiar_nombre(col):
                return re.sub(r"[><=].*", "", col).strip()

            lime_df["variable_limpia"] = lime_df["variable"].apply(limpiar_nombre)

            lime_df = lime_df[
                ~lime_df["variable_limpia"].str.startswith("state_")
                &
                ~lime_df["variable_limpia"].str.startswith("userchannel_")
            ]

            keys_validos = set(explicaciones.keys())
            lime_df = lime_df[lime_df["variable_limpia"].isin(keys_validos)]

            if lime_df.shape[0] == 0:
                st.info("No hay variables del diccionario que empujen claramente hacia churn.")
            else:
                lime_df["motivo"] = lime_df["variable_limpia"].map(explicaciones)

                top_motivos = (
                    lime_df.sort_values("peso", ascending=False)
                        .head(3)[["motivo", "peso"]]
                )

        


        #############

        # Guardar en session_state y archivo
        st.session_state["df_resultado"] = df_resultado_modelo
        df_resultado_modelo.to_parquet("predicciones_churn.parquet", index=True)
        top_motivos.to_parquet("top_motivos.parquet", index=True)
        #df_pred.to_parquet("df_pred.parquet", index=True)
        st.success("Predicciones generadas correctamente. Ya puedes ver las visualizaciones.")


# ============================================================
# 2. SI df_resultado EXISTE → MOSTRAR TABLA + VISUALIZACIONES
# ============================================================
df_resultado = st.session_state.get("df_resultado")

## no le el archivo??


with col_right:
    if df_resultado is not None:
        #df_proc = pd.read_parquet("df_pred.parquet")
        st.subheader("Resultados")

        df_subset = df_resultado[[
            "gender", "qualification", "usertype", "state",
            "age", "occupation", "prediccion_churn", "probabilidad_churn"
        ]]

        df_subset = df_subset[df_subset["prediccion_churn"] == 1]
        df_subset = df_subset.sort_values(
            by="probabilidad_churn", 
            ascending=False
        )

        st.dataframe(df_subset, use_container_width=True)


# ============================================================
# SI NO HAY df_resultado → DETENER STREAMLIT
# ============================================================
if df_resultado is None or df_resultado.empty:
    st.info("Carga un archivo y ejecuta el modelo para generar las predicciones.")
    st.stop()


# ============================================================
#   AQUI SIGUE TODO TU CÓDIGO DE VISUALIZACIONES SIN CAMBIOS
# ============================================================
# (Desde aquí puedes pegar todas tus gráficas exactamente como ya las tienes)

with st.sidebar:
    st.header("Filtros globales")

    # Listas únicas obtenidas de df_resultado
    gender_list = df_resultado["gender"].dropna().unique().tolist()
    qualification_list = df_resultado["qualification"].dropna().unique().tolist()
    usertype_list = df_resultado["usertype"].dropna().unique().tolist()
    state_list = df_resultado["state"].dropna().unique().tolist()

    f_gender = st.multiselect("Género", gender_list)
    f_qualification = st.multiselect("Nivel de cuenta", qualification_list)
    f_usertype = st.multiselect("Tipo de usuario", usertype_list)
    f_state = st.multiselect("Estado", state_list)

    # Filtro de edad basado en df_resultado
    min_age = int(df_resultado["age"].min())
    max_age = int(df_resultado["age"].max())

    f_age = st.slider(
        "Rango de Edad",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

# ============================================
# FUNCIÓN PARA APLICAR FILTROS AL DF_RESULTADO
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

# ============================================
# APLICACIÓN DE FILTROS
# ============================================
df_resultado_f = aplicar_filtros(df_resultado.copy())

# ============================================
#      VISUALIZACIONES
# ============================================
top_left, top_right = st.columns(2)

# ======================================================
# 3.1 DONA % CHURN
# ======================================================
with top_left:
    #st.markdown("#### % de usuarios con Churn (subset filtrado)")
    st.subheader("% de usuarios con Churn (subset filtrado)")
    counts = df_resultado_f["prediccion_churn"].value_counts()
    churn = counts.get(1, 0)
    nochurn = counts.get(0, 0)
    total = churn + nochurn

    if total > 0:
        pct_churn = churn / total * 100

        fig_donut = px.pie(
            values=[churn, nochurn],
            names=["Churn", "No Churn"],
            hole=0.55,
            color=["Churn", "No Churn"],
            color_discrete_map={
                "Churn": "#070A4A",
                "No Churn": "#31356E"
            }
        )

        fig_donut.update_layout(
            template="plotly_white",
            showlegend=True,
            annotations=[
                dict(
                    text=f"{pct_churn:.1f}%<br>({churn} usuarios)",
                    x=0.5, y=0.5,
                    font=dict(size=14, color="black"),
                    showarrow=False
                )
            ],
            margin=dict(t=10, b=10, l=10, r=10),
            height=330,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            legend_title=dict(font=dict(color="black")),
            legend=dict(font=dict(color="black"))
        )


        fig_donut.update_xaxes(
            color="black",
            tickfont=dict(color="black"),
            title_font=dict(size = 16,color="black"),
            gridcolor="#DDDDDD",
            zerolinecolor="black"
        )

        fig_donut.update_yaxes(
            color="black",
            tickfont=dict(color="black"),
            title_font=dict(size = 16,color="black"),
            gridcolor="#DDDDDD",
            zerolinecolor="black"
        )



        st.plotly_chart(fig_donut, use_container_width=True)

    else:
        st.info("No hay predicciones para calcular el porcentaje de churn.")

# ======================================================
# 3.2 MOTIVOS LIME
# ======================================================
with top_right:
    #st.markdown("#### Motivos principales de *Churn*")
    st.subheader("Motivos principales de Churn")
    top_motivos = pd.read_parquet("top_motivos.parquet")

    fig_mot = px.bar(
        top_motivos.sort_values("peso"),
        x="peso",
        y="motivo",
        orientation="h",
        color="peso",
        color_continuous_scale=[ "#A5B1EE", "#48549E", "#070A4A"]

    )

    fig_mot.update_layout(
        template="plotly_white",
        #title="Motivos principales de Churn",
        xaxis_title="Importancia (LIME)",
        yaxis_title="Motivo",
        margin=dict(t=40, l=10, r=10, b=10),
        height=330,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend_title=dict(font=dict(color="black")),
        legend=dict(font=dict(color="black"))
    )

    fig_mot.update_xaxes(
        color="black",
        tickfont=dict(color="black"),
        title_font=dict(size = 16,color="black"),
        gridcolor="#DDDDDD",
        zerolinecolor="black"
    )

    fig_mot.update_yaxes(
        color="black",
        tickfont=dict(color="black"),
        title_font=dict(size = 16,color="black"),
        gridcolor="#DDDDDD",
        zerolinecolor="black"
    )

    fig_mot.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
    fig_mot.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"))   


    fig_mot.update_coloraxes(
        colorbar=dict(
            title=dict(font=dict(color="black")),   # Título negro
            tickfont=dict(color="black")            # Valores negros
        )
)

    st.plotly_chart(fig_mot, use_container_width=True)


# ============================
# QUADRANTE INFERIOR
# ============================
bottom_left, bottom_right = st.columns(2)

# ======================================================
# 3.3 INGRESO EN RIESGO STACKED
# ======================================================
df_rev=None
with bottom_left:
    #st.markdown("#### Ingreso mensual en riesgo por nivel de probabilidad de *Churn*")
    st.subheader("Ingreso mensual en riesgo por nivel de probabilidad de Churn")
    if df_rev is not None:
        st.info("El archivo no tiene Revenue_dep y Revenue_ret para calcular el ingreso en riesgo.")
    else:
        df_risk = df_resultado_f.copy()
        # df_risk = df_rev.join(
        #     df_resultado[["prediccion_churn", "probabilidad_churn"]],
        #     how="inner"
        # )

        df_risk = df_risk[
            (df_risk["prediccion_churn"] == 1) &
            (df_risk["probabilidad_churn"] > 0.5)
        ]

        if df_risk.empty:
            st.info("No hay usuarios con churn y probabilidad > 0.5 para calcular ingreso en riesgo.")
        else:
            bin_edges  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
            bin_labels = ["50–60", "60–70", "70–80", "80–90", "90–100"]

            df_risk["prob_bin"] = pd.cut(
                df_risk["probabilidad_churn"],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
                right=False
            )

            risk_agg = (
                df_risk
                .groupby("prob_bin")[["Revenue_dep", "Revenue_ret"]]
                .sum()
                .reindex(bin_labels)
                .fillna(0.0)
            )

            total_revenue = risk_agg[["Revenue_dep", "Revenue_ret"]].sum().sum()

            if total_revenue == 0:
                st.info("No se pudo calcular ingreso en riesgo (revenues en cero).")
            else:
                fig_risk = go.Figure()

                fig_risk.add_bar(
                    x=risk_agg["Revenue_dep"],
                    y=risk_agg.index,
                    name="Depósitos",
                    orientation="h",
                    marker=dict(color="#48549E")
                )

                fig_risk.add_bar(
                    x=risk_agg["Revenue_ret"],
                    y=risk_agg.index,
                    name="Retiros",
                    orientation="h",
                    marker=dict(color="#A5B1EE")
                )

                fig_risk.update_layout(
                    template="plotly_white",
                    barmode="stack",
                    #title="Ingreso en riesgo por nivel de probabilidad de Churn",
                    xaxis_title="Ingreso mensual en riesgo (MXN)",
                    yaxis_title="Probabilidad de Churn (%)",
                    margin=dict(t=40, l=10, r=10, b=10),
                    height=330,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black"),
                    legend_title=dict(font=dict(color="black")),
                    legend=dict(font=dict(color="black"))
                )

                fig_risk.update_xaxes(
                    color="black",
                    tickfont=dict(color="black"),
                    title_font=dict(size = 16,color="black"),
                    gridcolor="#DDDDDD",
                    zerolinecolor="black"
                )

                fig_risk.update_yaxes(
                    color="black",
                    tickfont=dict(color="black"),
                    title_font=dict(size = 16,color="black"),
                    gridcolor="#DDDDDD",
                    zerolinecolor="black"
                )

                fig_risk.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
                fig_risk.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"))               

                st.plotly_chart(fig_risk, use_container_width=True)

# ======================================================
# 3.4 DISTRIBUCIÓN INGRESO HISTOGRAMA
# ======================================================
with bottom_right:
    #st.markdown("#### Distribución de ingreso mensual (Churn vs No churn)")
    st.subheader("Distribución de ingreso mensual (Churn vs No churn)")
    if df_rev is not None:
        st.info("El archivo no tiene Revenue_dep y Revenue_ret para calcular el ingreso mensual.")
    else:

        df_rev_hist = df_resultado_f.copy()

        df_rev_hist["Ingreso_total"] = df_rev_hist["Revenue_dep"] + df_rev_hist["Revenue_ret"]

        churn_vals = df_rev_hist[df_rev_hist["prediccion_churn"] == 1]["Ingreso_total"]
        nochurn_vals = df_rev_hist[df_rev_hist["prediccion_churn"] == 0]["Ingreso_total"]

        if churn_vals.empty or nochurn_vals.empty:
            st.info("No hay suficientes datos en ambos grupos para graficar el histograma.")
        else:
            fig_hist = go.Figure()

            fig_hist.add_histogram(
                x=nochurn_vals,
                nbinsx=20,
                name="No churn",
                opacity=0.6,
                marker=dict(color="#31356E")
            )

            fig_hist.add_histogram(
                x=churn_vals,
                nbinsx=20,
                name="Churn",
                opacity=0.8,
                marker=dict(color="#070A4A")
            )

            fig_hist.update_layout(
                template="plotly_white",
                barmode="overlay",
                #title="Distribución de ingreso mensual<br>Churn vs No churn",
                xaxis_title="Ingreso mensual estimado (MXN)",
                yaxis_title="Número de usuarios",
                margin=dict(t=40, l=10, r=10, b=10),
                height=330,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                legend_title=dict(font=dict(color="black")),
                legend=dict(font=dict(color="black"))
            )
            


            fig_hist.update_xaxes(
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(size = 16,color="black"),
                gridcolor="#DDDDDD",
                zerolinecolor="black"
            )

                        # ============================
                    # ============================
                        # DEFINIR LÍMITE X = Primer 30%
                        # ============================
            max_val = df_rev_hist["Ingreso_total"].max()
            x_limit = max_val * 0.30   # Primer 30% del rango

            fig_hist.update_xaxes(
                range=[0, x_limit],
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(size=16, color="black"),
                gridcolor="rgba(0,0,0,0.08)",    # gris MUY suave como la gráfica izquierda
                zerolinecolor="rgba(0,0,0,0.15)"  # más suave que negro
            )

            fig_hist.update_yaxes(
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(size=16, color="black"),
                gridcolor="rgba(0,0,0,0.08)",
                zerolinecolor="rgba(0,0,0,0.15)"
            )


            st.plotly_chart(fig_hist, use_container_width=True)




