import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


# -------------------------------------------------------
# Clase principal para el analisis del dataset
# -------------------------------------------------------

class DataAnalyzer:
    """
    Clase que encapsula las operaciones de analisis exploratorio
    sobre el dataset BankMarketing.
    """

    def __init__(self, df):
        self.df = df.copy()
        self.columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.columnas_categoricas = df.select_dtypes(include=["object"]).columns.tolist()

    def tabla_tipos(self):
        """Devuelve un DataFrame con el tipo, valores unicos y nulos por columna."""
        info = pd.DataFrame({
            "Columna": self.df.columns,
            "Tipo": [str(t) for t in self.df.dtypes.values],
            "Valores Unicos": [self.df[c].nunique() for c in self.df.columns],
            "Nulos": self.df.isnull().sum().values
        })
        return info

    def clasificar_variables(self):
        """Retorna un diccionario con las variables numericas y categoricas."""
        return {
            "numericas": self.columnas_numericas,
            "categoricas": self.columnas_categoricas
        }

    def estadisticas_descriptivas(self):
        """Retorna las estadisticas descriptivas de las variables numericas."""
        return self.df[self.columnas_numericas].describe().T

    def calcular_media(self, columna):
        return round(float(self.df[columna].mean()), 2)

    def calcular_mediana(self, columna):
        return round(float(self.df[columna].median()), 2)

    def calcular_moda(self, columna):
        return self.df[columna].mode()[0]

    def calcular_desviacion(self, columna):
        return round(float(self.df[columna].std()), 2)

    def conteo_unknowns(self):
        """Identifica la cantidad de valores 'unknown' por variable categorica."""
        resultados = []
        for col in self.columnas_categoricas:
            n = (self.df[col] == "unknown").sum()
            if n > 0:
                resultados.append({
                    "Variable": col,
                    "Cantidad": n,
                    "Porcentaje": round(n / len(self.df) * 100, 2)
                })
        return pd.DataFrame(resultados)

    def tasa_aceptacion(self):
        """Retorna conteo y porcentaje de la variable objetivo y."""
        conteo = self.df["y"].value_counts()
        porcentaje = self.df["y"].value_counts(normalize=True) * 100
        return pd.DataFrame({"Conteo": conteo, "Porcentaje (%)": porcentaje.round(2)})

    def grafico_histograma(self, columna, bins=30, separar_por_y=False):
        """Histograma con KDE para una variable numerica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        if separar_por_y:
            for val in self.df["y"].unique():
                datos = self.df[self.df["y"] == val][columna]
                sns.histplot(datos, kde=True, bins=bins, ax=ax, label=str(val), alpha=0.6)
            ax.legend(title="y")
        else:
            sns.histplot(self.df[columna], kde=True, bins=bins, ax=ax, color="steelblue")
        media = self.df[columna].mean()
        ax.axvline(media, color="red", linestyle="--", linewidth=1.4, label=f"Media = {media:.1f}")
        ax.legend()
        ax.set_title(f"Distribucion de {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        plt.tight_layout()
        return fig

    def grafico_barras(self, columna):
        """Grafico de barras horizontales para variable categorica."""
        fig, ax = plt.subplots(figsize=(9, 4))
        orden = self.df[columna].value_counts().index
        sns.countplot(data=self.df, y=columna, order=orden, ax=ax, color="steelblue")
        total = len(self.df)
        for p in ax.patches:
            pct = p.get_width() / total * 100
            ax.text(p.get_width() + total * 0.002,
                    p.get_y() + p.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=9)
        ax.set_title(f"Distribucion de {columna}")
        ax.set_xlabel("Conteo")
        ax.set_ylabel(columna)
        plt.tight_layout()
        return fig

    def grafico_boxplot(self, col_num, col_cat):
        """Boxplot y violinplot de una variable numerica agrupada por una categorica."""
        fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
        orden = self.df.groupby(col_cat)[col_num].median().sort_values(ascending=False).index
        sns.boxplot(data=self.df, x=col_cat, y=col_num, order=orden, palette="Set2", ax=ejes[0])
        ejes[0].set_title(f"{col_num} por {col_cat} — Boxplot")
        ejes[0].tick_params(axis="x", rotation=35)
        sns.violinplot(data=self.df, x=col_cat, y=col_num, order=orden, palette="Set3",
                       ax=ejes[1], inner="quartile")
        ejes[1].set_title(f"{col_num} por {col_cat} — Violin")
        ejes[1].tick_params(axis="x", rotation=35)
        plt.tight_layout()
        return fig

    def grafico_barras_apiladas(self, col1, col2):
        """Grafico de barras apiladas (porcentaje) para dos variables categoricas."""
        tabla = pd.crosstab(self.df[col1], self.df[col2], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        tabla.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.7)
        ax.set_title(f"{col1} vs {col2} — Porcentaje por grupo")
        ax.set_xlabel(col1)
        ax.set_ylabel("Porcentaje (%)")
        ax.legend(title=col2, bbox_to_anchor=(1.01, 1))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def grafico_correlacion(self):
        """Mapa de calor de correlacion entre variables numericas."""
        corr = self.df[self.columnas_numericas].corr()
        mascara = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mascara, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Matriz de Correlacion — Variables Numericas")
        plt.tight_layout()
        return fig

    def grafico_tasa_por_grupo(self, col_grupo):
        """Tasa de aceptacion (%) por cada categoria de una variable."""
        tasa = self.df.groupby(col_grupo)["y"].apply(
            lambda x: (x == "yes").sum() / len(x) * 100
        ).sort_values()
        promedio = tasa.mean()
        colores = ["#51cf66" if v >= promedio else "#ff6b6b" for v in tasa.values]
        fig, ax = plt.subplots(figsize=(8, max(3, len(tasa) * 0.45)))
        barras = ax.barh(tasa.index, tasa.values, color=colores, edgecolor="white")
        ax.axvline(promedio, color="navy", linestyle="--", linewidth=1.4,
                   label=f"Promedio: {promedio:.1f}%")
        for b in barras:
            ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
                    f"{b.get_width():.1f}%", va="center", fontsize=9)
        ax.legend()
        ax.set_xlabel("Tasa de Aceptacion (%)")
        ax.set_title(f"Tasa de Aceptacion por {col_grupo}")
        plt.tight_layout()
        return fig


# -------------------------------------------------------
# Configuracion de la pagina
# -------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="bank",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f0f4ff;
    border-radius: 8px;
    padding: 10px 14px;
    border-left: 4px solid #4472C4;
}
.caja-hallazgo {
    background-color: #e8f4ea;
    border-left: 4px solid #2e7d32;
    padding: 10px 14px;
    border-radius: 5px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Sidebar — menu de navegacion
# -------------------------------------------------------

with st.sidebar:
    st.title("Bank Marketing EDA")
    st.markdown("Analisis Exploratorio de Datos")
    st.markdown("---")
    modulo = st.radio(
        "Ir a:",
        ["Home", "Cargar Dataset", "Analisis EDA", "Conclusiones"]
    )
    st.markdown("---")
    st.markdown("**Tecnologias**")
    st.markdown("Python · Pandas · Seaborn · Streamlit")
    st.caption("Especializacion Python for Analytics · 2025")


# -------------------------------------------------------
# Modulo 1 — HOME
# -------------------------------------------------------

if modulo == "Home":

    st.title("Bank Marketing — Analisis Exploratorio de Datos")
    st.markdown("##### Caso de Estudio N°1 | Especializacion en Python for Analytics")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Objetivo del Analisis")
        st.markdown("""
Una institucion financiera ha experimentado una caida en la efectividad de sus campanas 
de marketing directo, pasando del **12% al 8%** en los ultimos seis meses. 
Esta situacion ha afectado el cumplimiento de metas de los ejecutivos comerciales.

El presente proyecto aplica un **Analisis Exploratorio de Datos (EDA)** sobre el dataset 
`BankMarketing.csv` para identificar los factores que influyen en la aceptacion o rechazo 
de la campana, con el objetivo de generar insights que apoyen la toma de decisiones.

El objetivo **no** es construir modelos predictivos, sino descubrir patrones y relaciones 
relevantes entre las variables disponibles.
        """)

        st.markdown("### Datos del Autor")
        st.info("""
Nombre: Miguel Angel Jimenez Huamani

Curso: Especializacion en Python for Analytics

        """)

    with col2:
        st.markdown("### Sobre el Dataset")
        st.markdown("""
**Archivo:** BankMarketing.csv

- 41 188 registros
- 21 variables
- Variable objetivo: `y` (yes / no)
- Campanas de marketing telefonico de un banco portugues
        """)

        st.markdown("### Tecnologias utilizadas")
        st.markdown("""
- Python 3.11
- Pandas y NumPy
- Matplotlib y Seaborn
- Streamlit
        """)

    st.divider()

    st.markdown("### Descripcion de Variables")
    variables = {
        "age": "Edad del cliente",
        "job": "Tipo de trabajo",
        "marital": "Estado civil",
        "education": "Nivel educativo",
        "default": "Tiene credito en mora",
        "housing": "Tiene credito hipotecario",
        "loan": "Tiene credito personal",
        "contact": "Canal de comunicacion",
        "month": "Mes del ultimo contacto",
        "day_of_week": "Dia del ultimo contacto",
        "duration": "Duracion de la llamada en segundos",
        "campaign": "Numero de contactos en la campana actual",
        "pdays": "Dias desde el ultimo contacto anterior",
        "previous": "Contactos antes de esta campana",
        "poutcome": "Resultado de la campana anterior",
        "emp.var.rate": "Tasa de variacion del empleo",
        "cons.price.idx": "Indice de precios al consumidor",
        "cons.conf.idx": "Indice de confianza del consumidor",
        "euribor3m": "Euribor a 3 meses",
        "nr.employed": "Numero de empleados",
        "y": "Variable objetivo: acepto la campana (yes/no)"
    }
    tabla_vars = pd.DataFrame(variables.items(), columns=["Variable", "Descripcion"])
    st.dataframe(tabla_vars, use_container_width=True, hide_index=True)


# -------------------------------------------------------
# Modulo 2 — CARGA DEL DATASET
# -------------------------------------------------------

elif modulo == "Cargar Dataset":

    st.title("Carga del Dataset")
    st.markdown("Sube el archivo **BankMarketing.csv** para habilitar el analisis.")

    archivo = st.file_uploader("Selecciona el archivo .csv", type=["csv"])

    if archivo is None:
        st.warning("Por favor sube el archivo CSV para continuar.")
        st.info("El archivo debe usar punto y coma (;) como separador.")
        st.stop()

    try:
        df = pd.read_csv(archivo, sep=";")
        st.session_state["df"] = df
        st.success(f"Archivo '{archivo.name}' cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", f"{df.shape[1]}")
    c3.metric("Variables Numericas", str(df.select_dtypes(include=["number"]).shape[1]))
    c4.metric("Variables Categoricas", str(df.select_dtypes(include=["object"]).shape[1]))

    st.markdown("### Vista previa del dataset")
    n_filas = st.slider("Cuantas filas mostrar", 5, 50, 10)
    st.dataframe(df.head(n_filas), use_container_width=True)

    st.markdown("### Nombres de las columnas")
    st.code(str(df.columns.tolist()))

    st.markdown("### Variable objetivo (y)")
    conteo_y = df["y"].value_counts()
    ca, cb = st.columns(2)
    ca.metric("Aceptaron (yes)", f"{conteo_y.get('yes', 0):,}")
    cb.metric("Rechazaron (no)", f"{conteo_y.get('no', 0):,}")
    st.caption(f"Tasa de conversion general: {conteo_y.get('yes', 0)/len(df)*100:.1f}%")


# -------------------------------------------------------
# Modulo 3 — ANALISIS EDA
# -------------------------------------------------------

elif modulo == "Analisis EDA":

    st.title("Analisis Exploratorio de Datos")

    if "df" not in st.session_state:
        st.warning("Primero ve al modulo 'Cargar Dataset' y sube el archivo CSV.")
        st.stop()

    df = st.session_state["df"]
    analizador = DataAnalyzer(df)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "1. Info General",
        "2. Clasificacion",
        "3. Estadisticas",
        "4. Valores Faltantes",
        "5. Dist. Numericas",
        "6. Dist. Categoricas",
        "7. Bivariado Num-Cat",
        "8. Bivariado Cat-Cat",
        "9. Analisis Dinamico",
        "10. Hallazgos"
    ])

    # --------------------------------------------------
    # Item 1 — Informacion general
    # --------------------------------------------------
    with tab1:
        st.subheader("Item 1: Informacion General del Dataset")
        st.markdown(f"""
El dataset contiene **{df.shape[0]:,} registros** y **{df.shape[1]} columnas** que describen 
las caracteristicas de clientes y el resultado de una campana de marketing telefonico.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Tipos de datos y valores unicos")
            st.dataframe(analizador.tabla_tipos(), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Valores nulos por columna")
            nulos = analizador.tabla_tipos()[["Columna", "Nulos"]]
            total_nulos = df.isnull().sum().sum()
            if total_nulos == 0:
                st.success("El dataset no presenta valores nulos.")
            else:
                st.dataframe(nulos[nulos["Nulos"] > 0], use_container_width=True, hide_index=True)

            st.markdown("#### Resumen de dimensiones")
            st.info(f"El dataset tiene **{df.shape[0]:,} filas** y **{df.shape[1]} columnas**.")

            resumen_tipo = df.dtypes.value_counts().reset_index()
            resumen_tipo.columns = ["Tipo de Dato", "Cantidad"]
            st.dataframe(resumen_tipo, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # Item 2 — Clasificacion de variables
    # --------------------------------------------------
    with tab2:
        st.subheader("Item 2: Clasificacion de Variables")

        clasificacion = analizador.clasificar_variables()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### Variables Numericas ({len(clasificacion['numericas'])})")
            for v in clasificacion["numericas"]:
                st.markdown(f"- `{v}`")
            st.success(f"Total: {len(clasificacion['numericas'])} variables numericas")

        with col2:
            st.markdown(f"#### Variables Categoricas ({len(clasificacion['categoricas'])})")
            for v in clasificacion["categoricas"]:
                st.markdown(f"- `{v}` — {df[v].nunique()} categorias")
            st.info(f"Total: {len(clasificacion['categoricas'])} variables categoricas")

        st.markdown("---")
        st.markdown("""
**Interpretacion:**
Las variables numericas incluyen indicadores demograficos como la edad, 
variables de campana como la duracion e indicadores macroeconomicos como el euribor y la tasa de empleo.
Las variables categoricas describen el perfil del cliente (trabajo, educacion, estado civil) 
y los canales y momentos de contacto.
La variable objetivo **`y`** es categorica binaria y registra si el cliente acepto la campana.
        """)

    # --------------------------------------------------
    # Item 3 — Estadisticas descriptivas
    # --------------------------------------------------
    with tab3:
        st.subheader("Item 3: Estadisticas Descriptivas")

        stats = analizador.estadisticas_descriptivas()
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        st.markdown("#### Metricas destacadas")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Edad Media", f"{analizador.calcular_media('age')} anios")
        c2.metric("Duracion Media", f"{analizador.calcular_media('duration')} seg")
        c3.metric("Contactos Promedio", str(analizador.calcular_media("campaign")))
        c4.metric("Euribor Promedio", str(analizador.calcular_media("euribor3m")))

        st.markdown("#### Analisis por variable")
        col1, col2 = st.columns(2)

        with col1:
            variable_sel = st.selectbox("Selecciona una variable", analizador.columnas_numericas,
                                        key="stat_sel")
            s = df[variable_sel]
            tabla_stat = pd.DataFrame({
                "Estadistico": ["Media", "Mediana", "Moda", "Desviacion Estandar",
                                "Minimo", "Q1 (25%)", "Q3 (75%)", "Maximo"],
                "Valor": [
                    round(s.mean(), 2),
                    round(s.median(), 2),
                    round(s.mode()[0], 2),
                    round(s.std(), 2),
                    round(s.min(), 2),
                    round(s.quantile(0.25), 2),
                    round(s.quantile(0.75), 2),
                    round(s.max(), 2)
                ]
            })
            st.dataframe(tabla_stat, use_container_width=True, hide_index=True)

        with col2:
            fig_stat, ax_stat = plt.subplots(figsize=(6, 4))
            sns.histplot(df[variable_sel], kde=True, color="steelblue", ax=ax_stat)
            ax_stat.axvline(s.mean(), color="red", linestyle="--", label="Media")
            ax_stat.axvline(s.median(), color="green", linestyle="--", label="Mediana")
            ax_stat.legend()
            ax_stat.set_title(f"Distribucion de {variable_sel}")
            st.pyplot(fig_stat)

        st.markdown(f"""
**Interpretacion:** Para la variable seleccionada, la diferencia entre la media 
({round(s.mean(), 2)}) y la mediana ({round(s.median(), 2)}) permite identificar 
el grado de asimetria de la distribucion.
        """)

    # --------------------------------------------------
    # Item 4 — Valores faltantes
    # --------------------------------------------------
    with tab4:
        st.subheader("Item 4: Analisis de Valores Faltantes")

        total_nulos = df.isnull().sum().sum()

        if total_nulos == 0:
            st.success("El dataset no presenta valores nulos en ninguna columna.")
        else:
            nulos_col = df.isnull().sum()
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(nulos_col[nulos_col > 0].rename("Nulos"), use_container_width=True)
            with col2:
                fig_nulos, ax_n = plt.subplots(figsize=(6, 3))
                nulos_col[nulos_col > 0].plot(kind="bar", ax=ax_n, color="coral")
                ax_n.set_title("Valores Nulos por Variable")
                st.pyplot(fig_nulos)

        st.markdown("---")
        st.markdown("#### Valores 'unknown' en variables categoricas")
        st.markdown("""
Aunque no hay nulos explicitos, el dataset contiene la etiqueta **'unknown'** 
en varias variables categoricas, que representa informacion no disponible del cliente.
        """)

        unknowns = analizador.conteo_unknowns()
        if not unknowns.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(unknowns, use_container_width=True, hide_index=True)
            with col2:
                fig_unk, ax_u = plt.subplots(figsize=(6, 4))
                ax_u.barh(unknowns["Variable"], unknowns["Porcentaje"], color="#e07b39",
                          edgecolor="white")
                ax_u.set_xlabel("Porcentaje con 'unknown' (%)")
                ax_u.set_title("Desconocidos por Variable (%)")
                plt.tight_layout()
                st.pyplot(fig_unk)

            st.warning("""
La variable 'default' presenta el mayor porcentaje de valores desconocidos (20.9%). 
Se recomienda tratar estos valores con la moda o crear una categoria propia en analisis futuros.
            """)

    # --------------------------------------------------
    # Item 5 — Distribucion de variables numericas
    # --------------------------------------------------
    with tab5:
        st.subheader("Item 5: Distribucion de Variables Numericas")

        col1, col2 = st.columns([1, 3])
        with col1:
            var_num = st.selectbox("Variable numerica", analizador.columnas_numericas, key="dist_num")
            bins = st.slider("Numero de bins", 10, 80, 30, key="bins_slider")
            sep_y = st.checkbox("Separar por variable objetivo (y)", value=False)
        with col2:
            fig_h = analizador.grafico_histograma(var_num, bins=bins, separar_por_y=sep_y)
            st.pyplot(fig_h)

        s_num = df[var_num]
        st.markdown(f"""
**Interpretacion de `{var_num}`:**
Media = {s_num.mean():.2f} | Mediana = {s_num.median():.2f} | Desviacion = {s_num.std():.2f}

La diferencia entre media y mediana indica el nivel de asimetria de la distribucion. 
Una desviacion estandar alta respecto a la media senala una distribucion muy dispersa.
        """)

        st.markdown("---")
        if st.checkbox("Ver todos los histogramas juntos"):
            n_cols_g = 3
            n_filas_g = int(np.ceil(len(analizador.columnas_numericas) / n_cols_g))
            fig_all, ejes = plt.subplots(n_filas_g, n_cols_g,
                                         figsize=(15, n_filas_g * 4))
            ejes = ejes.flatten()
            for i, col in enumerate(analizador.columnas_numericas):
                sns.histplot(df[col], kde=True, ax=ejes[i], color="steelblue")
                ejes[i].set_title(f"Distribucion de {col}")
            for j in range(len(analizador.columnas_numericas), len(ejes)):
                ejes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_all)

    # --------------------------------------------------
    # Item 6 — Distribucion de variables categoricas
    # --------------------------------------------------
    with tab6:
        st.subheader("Item 6: Analisis de Variables Categoricas")

        cats_sin_y = [c for c in analizador.columnas_categoricas if c != "y"]
        var_cat = st.selectbox("Variable categorica", cats_sin_y, key="cat_dist")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_bar = analizador.grafico_barras(var_cat)
            st.pyplot(fig_bar)
        with col2:
            conteo_cat = df[var_cat].value_counts()
            pct_cat = df[var_cat].value_counts(normalize=True) * 100
            tabla_cat = pd.DataFrame({"Conteo": conteo_cat, "Porcentaje (%)": pct_cat.round(2)})
            st.markdown(f"**Proporciones de `{var_cat}`**")
            st.dataframe(tabla_cat, use_container_width=True)

        st.markdown("---")
        st.markdown(f"**Tasa de aceptacion de la campana por `{var_cat}`**")
        fig_tasa = analizador.grafico_tasa_por_grupo(var_cat)
        st.pyplot(fig_tasa)

    # --------------------------------------------------
    # Item 7 — Bivariado: numerico vs categorico
    # --------------------------------------------------
    with tab7:
        st.subheader("Item 7: Analisis Bivariado — Numerico vs Categorico")
        st.markdown("""
Se compara la distribucion de una variable numerica segun los grupos 
de una variable categorica, para identificar si existen diferencias relevantes entre ellos.
        """)

        col1, col2 = st.columns(2)
        with col1:
            var_num_biv = st.selectbox("Variable Numerica", analizador.columnas_numericas,
                                       key="biv_num")
        with col2:
            var_cat_biv = st.selectbox(
                "Variable Categorica",
                ["y"] + [c for c in analizador.columnas_categoricas if c != "y"],
                key="biv_cat"
            )

        fig_biv = analizador.grafico_boxplot(var_num_biv, var_cat_biv)
        st.pyplot(fig_biv)

        agrupado = df.groupby(var_cat_biv)[var_num_biv].agg(["mean", "median", "std", "count"]).round(2)
        agrupado.columns = ["Media", "Mediana", "Desv. Estandar", "Registros"]
        st.markdown(f"**Estadisticas de `{var_num_biv}` por `{var_cat_biv}`**")
        st.dataframe(agrupado, use_container_width=True)

        st.info(f"""
La linea central en el boxplot representa la mediana de `{var_num_biv}` para cada grupo de `{var_cat_biv}`. 
Los puntos fuera de los bigotes son valores atipicos.
        """)

    # --------------------------------------------------
    # Item 8 — Bivariado: categorico vs categorico
    # --------------------------------------------------
    with tab8:
        st.subheader("Item 8: Analisis Bivariado — Categorico vs Categorico")
        st.markdown("""
Se analiza la relacion entre dos variables categoricas mediante una tabla de contingencia 
y un grafico de barras apiladas que muestra la proporcion de cada categoria.
        """)

        cats_sin_y2 = [c for c in analizador.columnas_categoricas if c != "y"]
        col1, col2 = st.columns(2)
        with col1:
            idx_edu = cats_sin_y2.index("education") if "education" in cats_sin_y2 else 0
            col_x = st.selectbox("Variable X (grupos)", cats_sin_y2, key="biv_cx",
                                 index=idx_edu)
        with col2:
            col_y2 = st.selectbox(
                "Variable Y (comparar con)",
                ["y"] + [c for c in cats_sin_y2 if c != col_x],
                key="biv_cy"
            )

        fig_cc = analizador.grafico_barras_apiladas(col_x, col_y2)
        st.pyplot(fig_cc)

        col_a, col_b = st.columns(2)
        with col_a:
            ct_conteos = pd.crosstab(df[col_x], df[col_y2])
            st.markdown("**Tabla de Contingencia — Conteos**")
            st.dataframe(ct_conteos, use_container_width=True)
        with col_b:
            ct_pct = pd.crosstab(df[col_x], df[col_y2], normalize="index") * 100
            st.markdown("**Tabla de Contingencia — Porcentaje por fila**")
            st.dataframe(ct_pct.round(1), use_container_width=True)

    # --------------------------------------------------
    # Item 9 — Analisis dinamico con filtros
    # --------------------------------------------------
    with tab9:
        st.subheader("Item 9: Analisis con Parametros Seleccionados")
        st.markdown("""
Usa los controles para filtrar el dataset y explorar subgrupos de interes. 
Los graficos y metricas se actualizan automaticamente.
        """)

        col1, col2 = st.columns(2)
        with col1:
            rango_edad = st.slider("Rango de Edad", int(df["age"].min()), int(df["age"].max()),
                                   (25, 60))
            sel_trabajo = st.multiselect("Tipo de Trabajo", df["job"].unique().tolist(),
                                         default=df["job"].unique().tolist()[:4])
        with col2:
            sel_edu = st.multiselect("Nivel Educativo", df["education"].unique().tolist(),
                                     default=df["education"].unique().tolist()[:3])
            cols_a_ver = st.multiselect("Variables a visualizar",
                                        analizador.columnas_numericas,
                                        default=["age", "duration", "campaign"])
            solo_yes = st.checkbox("Solo clientes que aceptaron (y = yes)")

        # Aplicar filtros
        filtro = (
            df["age"].between(*rango_edad) &
            df["job"].isin(sel_trabajo if sel_trabajo else df["job"].unique()) &
            df["education"].isin(sel_edu if sel_edu else df["education"].unique())
        )
        if solo_yes:
            filtro = filtro & (df["y"] == "yes")
        df_filtrado = df[filtro]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registros filtrados", f"{len(df_filtrado):,}")
        m2.metric("% del total", f"{len(df_filtrado)/len(df)*100:.1f}%")
        pct_yes_filt = (df_filtrado["y"] == "yes").mean() * 100 if len(df_filtrado) > 0 else 0
        m3.metric("Tasa de Aceptacion", f"{pct_yes_filt:.1f}%")
        edad_media_filt = f"{df_filtrado['age'].mean():.1f}" if len(df_filtrado) > 0 else "—"
        m4.metric("Edad Media", edad_media_filt)

        if len(df_filtrado) == 0:
            st.warning("No hay registros con los filtros aplicados. Ajusta los parametros.")
        else:
            if cols_a_ver:
                n_c_dyn = min(3, len(cols_a_ver))
                n_r_dyn = int(np.ceil(len(cols_a_ver) / n_c_dyn))
                fig_dyn, ejes_dyn = plt.subplots(n_r_dyn, n_c_dyn,
                                                  figsize=(5 * n_c_dyn, 4 * n_r_dyn))
                ejes_flat = np.array(ejes_dyn).flatten() if n_r_dyn * n_c_dyn > 1 else [ejes_dyn]
                for i, col in enumerate(cols_a_ver):
                    sns.histplot(df_filtrado[col], kde=True, ax=ejes_flat[i], color="steelblue",
                                 bins=25)
                    ejes_flat[i].set_title(f"{col} (datos filtrados)")
                for j in range(len(cols_a_ver), len(ejes_flat)):
                    ejes_flat[j].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_dyn)

            st.markdown("**Muestra de datos filtrados**")
            st.dataframe(df_filtrado.head(20), use_container_width=True)

    # --------------------------------------------------
    # Item 10 — Hallazgos clave
    # --------------------------------------------------
    with tab10:
        st.subheader("Item 10: Hallazgos Clave del Analisis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribucion de la Variable Objetivo")
            tasa = analizador.tasa_aceptacion()
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            ax_pie.pie(tasa["Conteo"], labels=tasa.index, autopct="%1.1f%%",
                       colors=["#e63946", "#51cf66"],
                       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
            ax_pie.set_title("Variable Objetivo y")
            st.pyplot(fig_pie)
            st.dataframe(tasa, use_container_width=True)

        with col2:
            st.markdown("#### Tasa de Aceptacion segun Campana Anterior")
            fig_pout = analizador.grafico_tasa_por_grupo("poutcome")
            st.pyplot(fig_pout)

        st.markdown("---")
        st.markdown("#### Matriz de Correlacion")
        fig_corr = analizador.grafico_correlacion()
        st.pyplot(fig_corr)

        st.markdown("---")
        st.markdown("#### Insights principales")

        hallazgos = [
            ("Clientes con exito previo tienen una tasa de conversion 6 veces mayor al promedio",
             "Los registros con poutcome = 'success' alcanzan tasas de aceptacion alrededor del 65%, "
             "frente al promedio general del 11.3%. Este es el segmento de mayor prioridad estrategica."),
            ("La duracion de la llamada es el indicador conductual mas fuerte",
             "Existe una diferencia clara en la duracion de las llamadas entre clientes que aceptan "
             "y los que rechazan. Una llamada mas larga refleja mayor interes del cliente."),
            ("Los segmentos etarios extremos responden mejor a la campana",
             "Los clientes menores de 30 y mayores de 60 anios superan el promedio de conversion, "
             "lo que sugiere estrategias diferenciadas segun el grupo de edad."),
            ("El entorno macroeconomico condiciona la efectividad",
             "Las variables euribor3m, emp.var.rate y nr.employed estan altamente correlacionadas "
             "entre si e impactan la probabilidad de aceptacion. Las campanas funcionan mejor "
             "cuando las tasas de interes son bajas."),
            ("El canal celular supera al telefono fijo en tasa de conversion",
             "El contacto via celular (contact = cellular) muestra tasas de aceptacion "
             "consistentemente superiores al canal telefonico tradicional.")
        ]

        for titulo, detalle in hallazgos:
            st.markdown(f"""
<div class="caja-hallazgo">
<b>{titulo}</b><br><small>{detalle}</small>
</div>""", unsafe_allow_html=True)


# -------------------------------------------------------
# Modulo 4 — CONCLUSIONES
# -------------------------------------------------------

elif modulo == "Conclusiones":

    st.title("Conclusiones Finales")
    st.markdown("""
A partir del analisis exploratorio realizado sobre el dataset BankMarketing.csv, 
se presentan cinco conclusiones orientadas a la toma de decisiones estrategicas.
    """)

    conclusiones = [
        {
            "titulo": "1. La duracion de la llamada refleja el nivel de interes del cliente",
            "texto": (
                "El analisis bivariado muestra que los clientes que aceptaron la campana tuvieron "
                "llamadas de mayor duracion en comparacion con quienes la rechazaron. "
                "Esto indica que la calidad de la conversacion es mas importante que la cantidad "
                "de contactos. Los ejecutivos deberian enfocarse en extender el dialogo con el cliente "
                "en lugar de realizar llamadas cortas y masivas, ya que una llamada larga "
                "esta directamente relacionada con el interes y la probabilidad de aceptacion."
            )
        },
        {
            "titulo": "2. El historial de campanas anteriores define la segmentacion de mayor valor",
            "texto": (
                "Los clientes con resultado exitoso en campanas previas (poutcome = success) "
                "presentan tasas de conversion que llegan al 65%, muy por encima del promedio "
                "del 11.3%. Este grupo representa el segmento de mayor rendimiento y deberia "
                "ser contactado en las primeras etapas de cada nueva campana. "
                "Construir y mantener una lista actualizada de estos clientes es una accion "
                "concreta y de bajo costo que puede impactar directamente la efectividad general."
            )
        },
        {
            "titulo": "3. Los extremos del rango etario presentan mayor propension a aceptar",
            "texto": (
                "Los clientes menores de 30 anios y mayores de 60 muestran tasas de aceptacion "
                "superiores al promedio, mientras que el grupo de 30 a 55 anios, que es el mas numeroso, "
                "resulta el menos receptivo. Esto sugiere desarrollar propuestas diferenciadas: "
                "productos de ahorro e inversion inicial para clientes jovenes, "
                "y productos de renta fija o depositos a plazo para adultos mayores, "
                "utilizando canales apropiados para cada perfil."
            )
        },
        {
            "titulo": "4. El contexto macroeconomico determina el momento optimo para lanzar campanas",
            "texto": (
                "Las variables euribor3m, emp.var.rate y nr.employed estan altamente correlacionadas "
                "entre si y con la tasa de aceptacion. Los periodos con euribor bajo coinciden "
                "con mayor predisposicion del cliente a depositar. "
                "El banco deberia monitorear estos indicadores y concentrar sus esfuerzos "
                "de campana en los momentos donde el entorno economico sea favorable, "
                "en lugar de distribuir los contactos de manera uniforme durante todo el anio."
            )
        },
        {
            "titulo": "5. El canal de contacto y el momento del mes influyen en la conversion",
            "texto": (
                "El canal celular (contact = cellular) muestra tasas de aceptacion "
                "consistentemente superiores al telefono fijo. Adicionalmente, "
                "el mes de mayo concentra la mayor cantidad de contactos sin un aumento "
                "proporcional en la tasa de aceptacion, lo que sugiere saturacion en ese periodo. "
                "Se recomienda migrar gradualmente al celular como canal principal "
                "y redistribuir los contactos hacia meses historicamente mas efectivos "
                "como marzo y diciembre."
            )
        }
    ]

    for c in conclusiones:
        with st.expander(f"**{c['titulo']}**", expanded=True):
            st.markdown(c["texto"])

    st.divider()

    st.markdown("### Recomendacion estrategica")
    st.success("""
Para revertir la caida en efectividad del 12% al 8%, se propone el siguiente plan de accion:

1. Contactar primero a clientes con campana anterior exitosa (poutcome = success)
2. Capacitar a ejecutivos en conversaciones de mayor duracion y calidad
3. Desarrollar propuestas diferenciadas para menores de 30 y mayores de 60 anios
4. Lanzar campanas en periodos de euribor bajo y confianza del consumidor alta
5. Migrar al canal celular como medio principal de contacto
    """)

    if "df" in st.session_state:
        df = st.session_state["df"]
        analizador = DataAnalyzer(df)
        st.divider()
        st.markdown("### Resumen visual")
        col1, col2 = st.columns(2)
        with col1:
            tasa = analizador.tasa_aceptacion()
            fig_res, ax_res = plt.subplots(figsize=(5, 4))
            ax_res.pie(tasa["Conteo"], labels=tasa.index, autopct="%1.1f%%",
                       colors=["#e63946", "#51cf66"], startangle=90,
                       wedgeprops={"edgecolor": "white", "linewidth": 2})
            ax_res.set_title("Distribucion variable objetivo y")
            st.pyplot(fig_res)
        with col2:
            fig_job = analizador.grafico_tasa_por_grupo("job")
            st.pyplot(fig_job)
