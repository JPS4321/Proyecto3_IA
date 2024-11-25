import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config(page_title="Agente Maestro de Juegos", page_icon="🎮", layout="wide")

    # Título e introducción
    st.title("🎮 Agente Maestro de Juegos con Streamlit")
    st.markdown(
        """
        Esta aplicación permite consultar información sobre los juegos más vendidos
        para Nintendo Switch, PlayStation 4 y Xbox One mediante un agente maestro.
        """
    )

    # Menú de selección de tareas predefinidas
    st.markdown("### Selección de tareas predefinidas")
    tasks = [
        "¿Cuál es el juego más vendido de Nintendo Switch?",
        "¿Cuál es el juego más vendido de PlayStation 4?",
        "¿Cuál es el juego más vendido de Xbox One?"
    ]
    selected_task = st.selectbox("Selecciona una tarea:", tasks)

    if st.button("Ejecutar tarea seleccionada"):
        process_predefined_task(selected_task)

    # Campo para preguntas generales
    st.markdown("### Preguntas sobre los juegos más vendidos")
    user_query = st.text_input("Escribe tu pregunta aquí:")

    if st.button("Procesar pregunta"):
        process_query(user_query)


# Función para procesar tareas predefinidas
def process_predefined_task(task):
    # Configuración de los agentes para cada consola
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    switch_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        path="switch.csv",  # Asegúrate de tener este archivo en el mismo directorio
        verbose=True,
        allow_dangerous_code=True
    )

    ps4_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        path="ps4.csv",  # Asegúrate de tener este archivo en el mismo directorio
        verbose=True,
        allow_dangerous_code=True
    )

    xbox_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        path="xboxone.csv",  # Asegúrate de tener este archivo en el mismo directorio
        verbose=True,
        allow_dangerous_code=True
    )

    # Asignar el agente adecuado según la tarea seleccionada
    if task == "¿Cuál es el juego más vendido de Nintendo Switch?":
        agent = switch_agent
    elif task == "¿Cuál es el juego más vendido de PlayStation 4?":
        agent = ps4_agent
    elif task == "¿Cuál es el juego más vendido de Xbox One?":
        agent = xbox_agent
    else:
        st.error("Tarea no reconocida.")
        return

    try:
        result = agent.invoke({"input": task})
        st.markdown("### Respuesta del agente:")
        st.code(result["output"], language="python")
    except Exception as e:
        st.error(f"Error al procesar la tarea: {str(e)}")


# Función para procesar preguntas generales
def process_query(query):
    if not query.strip():
        st.error("Por favor ingresa una pregunta válida.")
        return

    # Configuración de los agentes para cada consola
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    switch_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        path="switch.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    ps4_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        path="ps4.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    xbox_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        path="xboxone.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    # Crear herramientas para cada agente
    tools = [
        Tool(
            name="Nintendo Switch Agent",
            func=switch_agent.invoke,
            description="Responde preguntas sobre juegos de Nintendo Switch, como el más vendido."
        ),
        Tool(
            name="PlayStation 4 Agent",
            func=ps4_agent.invoke,
            description="Responde preguntas sobre juegos de PlayStation 4, como el más vendido."
        ),
        Tool(
            name="Xbox One Agent",
            func=xbox_agent.invoke,
            description="Responde preguntas sobre juegos de Xbox One, como el más vendido."
        ),
    ]

    # Crear el agente maestro
    grand_agent = create_react_agent(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        tools=tools,
        prompt=base_prompt.partial(instructions="Responde preguntas utilizando el agente adecuado.")
    )
    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True
    )

    try:
        # Procesar la consulta usando el agente maestro
        result = grand_agent_executor.invoke({"input": query})
        st.markdown("### Respuesta del agente maestro:")
        st.code(result["output"], language="python")
    except Exception as e:
        st.error(f"Error al procesar la pregunta: {str(e)}")


if __name__ == "__main__":
    main()
