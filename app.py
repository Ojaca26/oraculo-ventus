# app.py (versión con memoria y sin duplicados)

import streamlit as st
import pandas as pd
import re
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain

# ============================================
# 0) Configuración de la Página
# ============================================
st.set_page_config(page_title="Oráculo Ventus", page_icon="🔮", layout="wide")
st.title("🔮 Oráculo de Datos Ventus")
st.caption("Tu asistente IA para consultas y análisis de datos. Haz una pregunta y obtén respuestas al instante.")

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
            uri = f"mysql+pymymysql://{db_user}:{db_pass}@{db_host}/{db_name}"
            db = SQLDatabase.from_uri(uri, include_tables=["ventus", "ventus_rubro"])
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}", icon="🚨")
            st.stop()

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
            st.error(f"Error al inicializar los LLMs. Asegúrate de que tu API key es correcta. Error: {e}", icon="🚨")
            st.stop()

db = get_database_connection()
llm_sql, llm_analista, llm_orq = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    """Crea y cachea el agente SQL."""
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
    if df is None or df.empty: return "No hay datos en la tabla."
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)

def ejecutar_sql_real(pregunta_usuario: str):
    st.info("🤖 Traduciendo tu pregunta a consulta SQL...", icon="➡️")
    prompt = f"""
    Considerando la pregunta del usuario, genera una consulta SQL.
    Pregunta original: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt})
        st.code(sql_query.strip(), language='sql')
        with st.spinner("⏳ Ejecutando consulta en la base de datos..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query, "df": df, "texto": None}
    except Exception as e:
        st.warning(f"⚠️ La consulta directa falló. Intentando método alternativo... (Error: {e})", icon="⚙️")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str):
    st.info("🤔 Activando agente SQL experto como plan B...", icon="➡️")
    prompt = f"""
    Responde consultando la BD. Devuelve un resultado legible en tabla/resumen.
    Pregunta: "{pregunta_usuario}"
    """
    try:
        with st.spinner("💬 Consultando con el agente experto..."):
            res = agente_sql.invoke(prompt)
            texto = res["output"] if isinstance(res, dict) else str(res)
        
        df_md = markdown_table_to_df(texto)
        
        # MODIFICACIÓN: Si logramos parsear un DF, no devolvemos el texto original.
        if not df_md.empty:
            st.success("✅ Agente experto obtuvo y procesó los datos.")
            return {"texto": None, "df": df_md}
        else:
            st.warning("⚠️ El agente experto respondió, pero no en formato de tabla.", icon="📝")
            return {"texto": texto, "df": pd.DataFrame()}
            
    except Exception as e:
        st.error(f"❌ El agente SQL experto también falló: {e}", icon="🚨")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

def analizar_con_datos(pregunta_usuario: str, df: pd.DataFrame):
    st.info("🧠 Analista experto examinando los datos...", icon="➡️")
    df_resumen = _df_preview(df)
    prompt_analisis = f"""
    Eres un analista de datos senior para la empresa VENTUS.
    Pregunta del usuario: "{pregunta_usuario}"

    Basado en la siguiente tabla de datos:
    {df_resumen}

Análisis Ejecutivo de Datos,
Cuando recibas una tabla de resultados (facturación, ventas, métricas, etc.), realiza el siguiente análisis:
1. Calcular totales y porcentajes clave (participación de facturas grandes, distribución por días, % acumulado).
2. Detectar concentración (si pocos registros explican gran parte del total).
3. Identificar patrones temporales (días o periodos con concentración inusual).
4. Analizar dispersión (ticket promedio, comparación entre valores grandes vs pequeños).

Entregar el resultado en 3 bloques:
📌 Resumen Ejecutivo: hallazgos principales con números.
🔍 Números de referencia: totales, promedios, ratios comparativos.

⚠ Importante: No describas lo obvio de la tabla. Sé muy breve, directo y diciente, con frases cortas en bullets que un gerente pueda leer en 1 minuto.
    """
    with st.spinner("💡 Generando análisis y recomendaciones..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def clasificar_intencion(pregunta: str) -> str:
    prompt_orq = f"""
    Devuelve UNA sola palabra: `consulta` (si pide datos) o `analista` (si pide interpretar/analizar).
    Mensaje: "{pregunta}"
    """
    clasificacion = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
    return "analista" if "analista" in clasificacion else "consulta"

def obtener_datos_sql(pregunta_usuario: str) -> dict:
    res_real = ejecutar_sql_real(pregunta_usuario)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario)

def orquestador(pregunta_usuario: str, historial_chat: list):
    with st.expander("⚙️ Ver Proceso del Agente", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        with st.spinner("🔍 Analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ Tarea detectada: {clasificacion.upper()}.")

        resultado = {"tipo": clasificacion, "df": None, "analisis": None, "texto": None}

        if clasificacion == "analista":
            # MODIFICACIÓN (MEMORIA): Buscar la última tabla en el historial
            df_en_memoria = None
            if historial_chat:
                ultima_respuesta = historial_chat[-1]
                if ultima_respuesta["role"] == "assistant" and "df" in ultima_respuesta["content"]:
                    df_previo = ultima_respuesta["content"]["df"]
                    if isinstance(df_previo, pd.DataFrame) and not df_previo.empty:
                        df_en_memoria = df_previo
                        st.info("🧠 Usando la última tabla mostrada para el análisis.", icon="💾")

            if df_en_memoria is not None:
                analisis = analizar_con_datos(pregunta_usuario, df_en_memoria)
                resultado["analisis"] = analisis
                resultado["df"] = df_en_memoria # También devolvemos el DF para mostrarlo de nuevo
            else:
                st.warning("⚠️ No encontré una tabla reciente. Buscaré los datos de nuevo.", icon="🔍")
                res_datos = obtener_datos_sql(pregunta_usuario)
                resultado.update(res_datos)
                if res_datos.get("df") is not None and not res_datos["df"].empty:
                    analisis = analizar_con_datos(pregunta_usuario, res_datos["df"])
                    resultado["analisis"] = analisis
                else:
                    resultado["texto"] = "No pude obtener datos para analizar."
        
        else: # Si es 'consulta'
            res_datos = obtener_datos_sql(pregunta_usuario)
            resultado.update(res_datos)
    
    return resultado

# ============================================
# 3) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if content.get("texto"):
            st.markdown(content["texto"])
        if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty:
            st.dataframe(content["df"])
        if content.get("analisis"):
            st.markdown(content["analisis"])

# Input del usuario
if prompt := st.chat_input("Ej: 'Muéstrame la facturación total por rubro'"):
    # Guardar y mostrar pregunta del usuario
    user_message = {"role": "user", "content": {"texto": prompt}}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        res = orquestador(prompt, st.session_state.messages)
        
        # MODIFICACIÓN: Lógica de visualización simplificada
        response_content = {}
        
        if res.get("analisis"):
            st.markdown(res["analisis"])
            response_content["analisis"] = res["analisis"]
            # Si hay análisis de una tabla, la volvemos a mostrar para dar contexto
            if isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                 st.dataframe(res["df"])
                 response_content["df"] = res["df"]

        elif isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
            st.dataframe(res["df"])
            response_content["df"] = res["df"]
        
        elif res.get("texto"):
            st.markdown(res["texto"])
            response_content["texto"] = res["texto"]
        
        # Guardar respuesta del asistente en el historial
        if response_content:
            assistant_message = {"role": "assistant", "content": response_content}
            st.session_state.messages.append(assistant_message)
