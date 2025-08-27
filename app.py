# app.py (versión final con separador de tareas para evitar alucinaciones)

import streamlit as st
import pandas as pd
import re
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain

# ============================================
# 0) Configuración de la Página
# ============================================
st.set_page_config(page_title="Oráculo Ventus", page_icon="🔮", layout="wide")
st.title("🔮 Oráculo de Datos Ventus")
st.caption("Tu asistente IA para consultas y análisis de datos. Haz una pregunta y obtén respuestas al instante.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs
# ============================================

@st.cache_resource
def get_database_connection():
    # ... (Sin cambios)
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
            st.error(f"Error al conectar a la base de datos: {e}", icon="🚨")
            st.stop()

@st.cache_resource
def get_llms():
    # ... (Sin cambios)
    with st.spinner("🧠 Inicializando modelos de IA..."):
        try:
            api_key = st.secrets["google_api_key"]
            llm_sql = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            llm_analista = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
            llm_orq = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            st.success("✅ Modelos de IA listos.")
            return llm_sql, llm_analista, llm_orq
        except Exception as e:
            st.error(f"Error al inicializar los LLMs: {e}", icon="🚨")
            st.stop()

db = get_database_connection()
llm_sql, llm_analista, llm_orq = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    # ... (Sin cambios)
    with st.spinner("🛠️ Configurando agente SQL..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False)
        st.success("✅ Agente SQL configurado.")
        return agent

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 2) Funciones de Agentes (Lógica Principal con Mejoras)
# ============================================

def markdown_table_to_df(texto: str) -> pd.DataFrame:
    # ... (Sin cambios)
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
    # ... (Sin cambios)
    if df is None or df.empty: return "No hay datos en la tabla."
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)

def limpiar_y_extraer_sql(texto_con_sql: str) -> str:
    # ... (Sin cambios)
    match = re.search(r"```sql(.*?)```|SELECT.*?;", texto_con_sql, re.DOTALL | re.IGNORECASE)
    if match:
        sql_puro = match.group(1) if match.group(1) else match.group(0)
        return sql_puro.strip()
    return texto_con_sql

# ========== NUEVA FUNCIÓN "GERENTE" ==========
def extraer_pregunta_para_sql(pregunta_compleja: str) -> str:
    """Toma una pregunta de análisis y la convierte en una pregunta para la base de datos."""
    st.info("🧠 Reformulando la pregunta para el agente de SQL...", icon="🤔")
    prompt_extractor = f"""
    A partir de la siguiente solicitud de un usuario, extrae únicamente la pregunta específica sobre los datos que se necesita para responderla.
    Tu objetivo es crear una pregunta clara y concisa que pueda ser respondida con una consulta SQL.
    Por ejemplo, si el usuario pide "Analiza el comportamiento de la facturación de este año", tu deberías extraer "dame la facturación de este año".
    Si pide "Cuál es la tendencia de ventas del último trimestre", extrae "dame las ventas del último trimestre".

    Solicitud original: "{pregunta_compleja}"
    
    Extrae la pregunta para la base de datos:
    """
    pregunta_extraida = llm_orq.invoke(prompt_extractor).content.strip()
    st.info(f"📝 Pregunta para SQL: '{pregunta_extraida}'")
    return pregunta_extraida

def ejecutar_sql_real(pregunta_usuario: str):
    # ... (Sin cambios)
    st.info("🤖 Traduciendo tu pregunta a consulta SQL...", icon="➡️")
    prompt = f"""
    Considerando la pregunta del usuario, genera una consulta SQL para la base de datos MariaDB.
    IMPORTANTE:
    1. NUNCA limites los resultados (NO uses LIMIT) a menos que el usuario pida explícitamente un número pequeño de filas.
    2. Si la pregunta es abierta, asume que el usuario quiere ver todos los registros relevantes.
    Pregunta original: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        respuesta_llm = query_chain.invoke({"question": prompt})
        sql_query = limpiar_y_extraer_sql(respuesta_llm)
        st.code(sql_query, language='sql')
        with st.spinner("⏳ Ejecutando consulta en la base de datos..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query, "df": df, "texto": None}
    except Exception as e:
        st.warning(f"⚠️ La consulta directa falló. Intentando método alternativo... (Error: {e})", icon="⚙️")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str):
    # ... (Sin cambios)
    st.info("🤔 Activando agente SQL experto como plan B...", icon="➡️")
    prompt = f'Responde consultando la BD. Devuelve un resultado legible en tabla/resumen. Pregunta: "{pregunta_usuario}"'
    try:
        with st.spinner("💬 Consultando con el agente experto..."):
            res = agente_sql.invoke(prompt)
            texto = res["output"] if isinstance(res, dict) else str(res)
        df_md = markdown_table_to_df(texto)
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
    # ... (Sin cambios)
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
    # ... (Sin cambios)
    prompt_orq = f'Devuelve UNA sola palabra: `consulta` (si pide datos) o `analista` (si pide interpretar/analizar). Mensaje: "{pregunta}"'
    clasificacion = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
    return "analista" if "analista" in clasificacion else "consulta"

def obtener_datos_sql(pregunta_usuario: str) -> dict:
    # ... (Sin cambios)
    res_real = ejecutar_sql_real(pregunta_usuario)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario)

# ========== ORQUESTADOR MODIFICADO ==========
def orquestador(pregunta_usuario: str, historial_chat: list):
    with st.expander("⚙️ Ver Proceso del Agente", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        with st.spinner("🔍 Analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ Tarea detectada: {clasificacion.upper()}.")

        resultado = {"tipo": clasificacion, "df": None, "analisis": None, "texto": None}

        if clasificacion == "analista":
            df_en_memoria = None
            if historial_chat:
                for mensaje_previo in reversed(historial_chat):
                    if mensaje_previo["role"] == "assistant" and "df" in mensaje_previo["content"]:
                        df_previo = mensaje_previo["content"]["df"]
                        if isinstance(df_previo, pd.DataFrame) and not df_previo.empty:
                            df_en_memoria = df_previo
                            st.info("🧠 Usando la última tabla mostrada para el análisis.", icon="💾")
                            break
            
            if df_en_memoria is not None:
                # Si hay tabla en memoria, la usamos directamente para el análisis
                analisis = analizar_con_datos(pregunta_usuario, df_en_memoria)
                resultado["analisis"] = analisis
                resultado["df"] = df_en_memoria
            else:
                # Si NO hay tabla en memoria, usamos la nueva función "gerente"
                pregunta_sql = extraer_pregunta_para_sql(pregunta_usuario)
                res_datos = obtener_datos_sql(pregunta_sql)
                resultado.update(res_datos)
                
                if res_datos.get("df") is not None and not res_datos["df"].empty:
                    # Usamos la pregunta original del usuario para el análisis, no la extraída
                    analisis = analizar_con_datos(pregunta_usuario, res_datos["df"])
                    resultado["analisis"] = analisis
                else:
                    resultado["texto"] = "No pude obtener los datos necesarios para realizar el análisis."
        
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
    # ... (Sin cambios)
    with st.chat_message(message["role"]):
        content = message["content"]
        if "pregunta" in content:
            st.markdown(content["pregunta"])
        if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty:
            st.dataframe(content["df"])
        if content.get("analisis"):
            st.markdown(content["analisis"])
        if content.get("texto_plano"):
            st.markdown(content["texto_plano"])

# Input del usuario
if prompt := st.chat_input("Ej: 'Muéstrame la facturación total por rubro'"):
    # ... (Sin cambios en esta sección)
    user_message = {"role": "user", "content": {"pregunta": prompt}}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        res = orquestador(prompt, st.session_state.messages)
        
        response_content = {}
        
        if res.get("analisis"):
            st.markdown(res["analisis"])
            response_content["analisis"] = res["analisis"]
            if isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                 st.dataframe(res["df"])
                 response_content["df"] = res["df"]
        elif isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
            st.dataframe(res["df"])
            response_content["df"] = res["df"]
        elif res.get("texto"):
            st.markdown(res["texto"])
            response_content["texto_plano"] = res["texto"]
        
        if response_content:
            assistant_message = {"role": "assistant", "content": response_content}
            st.session_state.messages.append(assistant_message)
