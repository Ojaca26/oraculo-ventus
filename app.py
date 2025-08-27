# app.py

import streamlit as st
import pandas as pd
import re
import time
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
import plotly.express as px

# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="Agente IA para Análisis de Datos", page_icon="🤖", layout="wide")
st.title("🤖 Agente IA para Análisis de Datos con Gemini")
st.caption("Escribe una pregunta sobre tus datos y el agente se encargará de consultarla, analizarla y visualizarla.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs (con caché para eficiencia)
# ============================================

@st.cache_resource
def get_database_connection():
    """Establece y cachea la conexión a la base de datos."""
    with st.spinner("🔌 Conectando a la base de datos..."):
        try:
            db_user = st.secrets["db_credentials"]["user"]
            db_pass = st.secrets["db_credentials"]["password"]
            db_host = st.secrets["db_credentials"]["host"]
            db_name = st.secrets["db_credentials"]["database"]
            uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"
            db = SQLDatabase.from_uri(uri, include_tables=["ventus", "ventus_rubro"])
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    """Inicializa y cachea los modelos de lenguaje."""
    with st.spinner("🧠 Inicializando modelos de IA (esto puede tardar un momento)..."):
        try:
            api_key = st.secrets["google_api_key"]
            llm_sql = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.1, google_api_key=api_key)
            llm_analista = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
            llm_orq = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            st.success("✅ Modelos de IA listos.")
            return llm_sql, llm_analista, llm_orq
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Asegúrate de que tu API key es correcta. Error: {e}")
            return None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    """Crea y cachea el agente SQL."""
    if not _llm or not _db:
        return None
    with st.spinner("🛠️ Configurando agente SQL..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False)
        st.success("✅ Agente SQL configurado.")
        return agent

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 2) Funciones de Agentes (Lógica Principal)
# ============================================

def markdown_table_to_df(texto: str) -> pd.DataFrame:
    # ... (El código de esta función es idéntico al original)
    lineas = [l.strip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        s = df[c].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
        try: df[c] = pd.to_numeric(s)
        except Exception: df[c] = s
    return df

def _df_preview(df: pd.DataFrame, n: int = 20) -> str:
    # ... (El código de esta función es idéntico al original)
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)

# --- Funciones de ejecución modificadas para Streamlit ---

def ejecutar_sql_real(pregunta_usuario: str):
    st.info("🤖 Entendido. Estoy traduciendo tu pregunta a una consulta SQL...")
    prompt_con_instrucciones = f"""
    Considerando la pregunta del usuario, genera una consulta SQL.
    IMPORTANTE: Si agregas o calculas una columna, usa uno de los siguientes alias estándar:
    - Para valores monetarios o conteos: 'total_valor'
    - Para fechas o periodos: 'fecha'
    - Para categorías (rubros, proveedores, etc.): 'categoria'
    Pregunta original: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt_con_instrucciones})
        st.code(sql_query.strip(), language='sql')
        with st.spinner("⏳ Ejecutando la consulta en la base de datos..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando un método alternativo... Error: {e}")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str):
    st.info("🤔 La consulta directa falló. Activando mi agente SQL experto como plan B.")
    prompt_sql = (
        "Responde consultando la BD. Devuelve un resultado legible en tabla/resumen, "
        "siempre organiza cronológicamente la información si esta dada en fechas, "
        "sin explicar pasos internos. Limita a 200 filas si aplica. Pregunta: "
        f"{pregunta_usuario}"
    )
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda en lenguaje natural..."):
            res = agente_sql.invoke(prompt_sql)
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        st.info("📝 Recibí una respuesta en texto. Intentando convertirla en una tabla de datos...")
        df_md = markdown_table_to_df(texto)
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

def analizar_con_datos(pregunta_usuario: str, datos_texto: str, df: pd.DataFrame | None):
    st.info("\n🧠 Ahora, mi analista experto está examinando los datos para encontrar insights clave...")
    df_resumen = _df_preview(df, 20)
    prompt_analisis = f"""
    Eres un analista senior, trabajas para la empresa VENTUS... (el resto del prompt es idéntico)
    Pregunta del usuario: {pregunta_usuario}
    Datos/Resultados disponibles:
    TEXTO: {datos_texto}
    TABLA (primeras filas): {df_resumen}
    Análisis Ejecutivo de Datos... (el resto del prompt es idéntico)
    """
    with st.spinner("💡 Generando análisis y recomendaciones..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def agente_visualizador(pregunta: str, df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("⚠️ No hay datos para graficar.")
        return
    
    st.info("\n🎨 Analizando los datos para encontrar la mejor forma de visualizarlos...")
    time.sleep(1)

    # Lógica de detección de columnas (simplificada del original)
    def _col(df, *nombres):
        for n in nombres:
            if n in df.columns: return n
        # Búsqueda insensible a mayúsculas como fallback
        for n in nombres:
            for col_name in df.columns:
                if n.lower() == col_name.lower(): return col_name
        return None

    df2 = df.copy()
    col_fecha = _col(df2, "fecha", "Fecha_aprobacion", "Fecha", "date")
    col_valor = _col(df2, "total_valor", "Total_COP", "total_cop", "Total_USD", "monto", "valor", "Total", "total")
    col_cat = _col(df2, "categoria", "Rubro_CF", "rubro", "Proveedor", "proveedor", "Categoria", "categoría", "Rubro", "rubro_descripcion")

    if col_fecha:
        try: df2[col_fecha] = pd.to_datetime(df2[col_fecha], errors='coerce')
        except: pass

    text_hint = (pregunta or "").lower()
    fig = None

    if any(k in text_hint for k in ["línea", "evolución", "tiempo"]) and col_fecha and col_valor:
        st.info("📈 He decidido que un gráfico de líneas es la mejor opción...")
        fig = px.line(df2.sort_values(by=col_fecha), x=col_fecha, y=col_valor, color=col_cat, title="Evolución Temporal")
    elif col_valor and col_cat:
        st.info("📊 Un gráfico de barras será ideal para comparar las categorías...")
        dfx = df2.groupby(col_cat, as_index=False)[col_valor].sum().sort_values(col_valor, ascending=False).head(25)
        fig = px.bar(dfx, x=col_cat, y=col_valor, title="Comparativo por Categoría")
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ No pude determinar columnas adecuadas para graficar. Columnas disponibles: {df.columns.tolist()}")

# --- Orquestador Principal ---

def clasificar_intencion(pregunta: str) -> str:
    # ... (El código de esta función es idéntico al original, pero sin prints)
    prompt_orq = f"""
    Devuelve UNA sola palabra exacta según la intención:
    - `consulta` -> si el usuario pide extraer/filtrar/contar datos.
    - `analista` -> si el usuario pide interpretar y recomendar acciones.
    - `visualizador` -> si el usuario pide gráfico, visual, comparar en gráfico, heatmap, línea de tiempo, dashboard, mapa.
    Mensaje: {pregunta}
    """
    clasificacion = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
    return clasificacion

def obtener_datos_sql(pregunta_usuario: str) -> dict:
    res_real = ejecutar_sql_real(pregunta_usuario)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return {"sql": res_real["sql"], "df": res_real["df"], "texto": None}
    res_nat = ejecutar_sql_en_lenguaje_natural(pregunta_usuario)
    return {"sql": None, "df": res_nat["df"], "texto": res_nat["texto"]}

def orquestador(pregunta_usuario: str):
    with st.expander("⚙️ Ver Proceso del Agente", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        with st.spinner("🔍 Analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        res_datos = obtener_datos_sql(pregunta_usuario)
        
        resultado = {"tipo": clasificacion, **res_datos, "analisis": None}
        
        if "analista" in clasificacion:
            if res_datos.get("df") is not None and not res_datos["df"].empty:
                analisis = analizar_con_datos(pregunta_usuario, res_datos.get("texto", ""), res_datos["df"])
                resultado["analisis"] = analisis
            else:
                st.warning("No se pudieron obtener datos, por lo que no se puede realizar el análisis.")
                resultado["analisis"] = "No se pudo generar un análisis porque no se obtuvieron datos."

    return resultado

# ============================================
# 3) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Itera sobre el contenido del mensaje para mostrarlo correctamente
        if "texto" in message["content"]: st.markdown(message["content"]["texto"])
        if "df" in message["content"] and message["content"]["df"] is not None: st.dataframe(message["content"]["df"])
        if "analisis" in message["content"] and message["content"]["analisis"]: st.markdown(message["content"]["analisis"])
        if "fig" in message["content"] and message["content"]["fig"]: st.plotly_chart(message["content"]["fig"], use_container_width=True)


if prompt := st.chat_input("¿Qué quieres saber de tus datos?"):
    if not all([db, llm_sql, llm_analista, llm_orq, agente_sql]):
        st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
    else:
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar respuesta del asistente
        with st.chat_message("assistant"):
            res = orquestador(prompt)
            
            # Mostrar los resultados principales
            st.markdown(f"### Resultado para: '{prompt}'")
            if res.get("df") is not None and not res["df"].empty:
                st.dataframe(res["df"])
            elif res.get("texto"):
                 st.markdown(res["texto"])
            
            if res.get("analisis"):
                st.markdown("---")
                st.markdown("### 🧠 Análisis Experto")
                st.markdown(res["analisis"])
                
            if "visualizador" in res["tipo"]:
                st.markdown("---")
                st.markdown("### 📊 Visualización")
                agente_visualizador(prompt, res["df"])

            # Guardar la respuesta completa en el historial (simplificado para evitar guardar figuras)
            response_to_save = {k: v for k, v in res.items() if k != 'fig'}
            st.session_state.messages.append({"role": "assistant", "content": response_to_save})