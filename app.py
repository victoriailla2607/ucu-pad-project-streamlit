import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="California Housing", layout="wide")


# ---------- Fase 1: carga y preparación ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target
    return df


df_california = load_data()

st.title("Análisis interactivo: California Housing")
st.caption("Dataset: sklearn.datasets.fetch_california_housing (censo 1990). "
           "MedHouseVal está en cientos de miles de USD.")

st.header("1. Datos")
st.write("**Valores faltantes por columna:**")
st.dataframe(df_california.isna().sum().rename("faltantes"))
st.write("**Primeras 5 filas:**")
st.dataframe(df_california.head())
st.write("**Tipo de dato de cada columna:**")
st.dataframe(df_california.dtypes.astype(str).rename("dtype"))

# ---------- Fase 2: filtros ----------
st.sidebar.markdown("## Filtros")
st.sidebar.markdown("Ajustá los controles para filtrar los grupos de bloques. "
                    "Los resúmenes y gráficos se actualizan solos.")

age_min = float(df_california["HouseAge"].min())
age_max = float(df_california["HouseAge"].max())
age_range = st.sidebar.slider(
    "Edad mediana de la vivienda (HouseAge)",
    min_value=age_min, max_value=age_max, value=(age_min, age_max),
)

lat_min = float(df_california["Latitude"].min())
lat_max = float(df_california["Latitude"].max())
min_lat = st.sidebar.number_input(
    "Latitud mínima",
    min_value=lat_min, max_value=lat_max, value=lat_min, step=0.1,
)

filtered = df_california[
    df_california["HouseAge"].between(age_range[0], age_range[1])
    & (df_california["Latitude"] >= min_lat)
]

st.header("2. Resumen descriptivo (datos filtrados)")
st.write(f"Registros: **{len(filtered):,}** de {len(df_california):,}")

if filtered.empty:
    st.warning("No hay datos con los filtros actuales. Ampliá los rangos.")
    st.stop()

val = filtered["MedHouseVal"]
c1, c2 = st.columns(2)
c1.metric("Mediana de MedHouseVal", f"{val.median():.3f}")
c2.metric("Rango (máx − mín)", f"{val.max() - val.min():.3f}")

# ---------- Fase 3: visualización ----------
st.header("3. Visualización")

fig_hist = px.histogram(
    filtered, x="MedHouseVal", nbins=40,
    title="Distribución del valor mediano de la vivienda",
    labels={"MedHouseVal": "Valor mediano (cientos de miles USD)"},
)
st.plotly_chart(fig_hist, use_container_width=True)

fig_scatter = px.scatter(
    filtered, x="MedInc", y="MedHouseVal", opacity=0.4,
    title="Ingreso mediano vs. valor mediano de la vivienda",
    labels={"MedInc": "Ingreso mediano (decenas de miles USD)",
            "MedHouseVal": "Valor mediano (cientos de miles USD)"},
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Mapa (opcional)")
st.map(filtered, latitude="Latitude", longitude="Longitude")
