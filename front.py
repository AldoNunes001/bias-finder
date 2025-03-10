import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import loaders

CHAT_MEMORY = ConversationBufferMemory()

VALID_FILE_TYPES = ["PDF", "TXT"]

MODELS_CONFIG = {
    "Groq": {
        "models": ["deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b",
                   "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "models": ["chatgpt-4o-latest", "gpt-4o-mini", "o1", "o1-mini"],
        "chat": ChatOpenAI,
    },
}

def load_file(file_type, file):
    if file_type == "PDF":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        document = loaders.load_pdf(temp_file_path)

    elif file_type == "TXT":
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        document = loaders.load_txt(temp_file_path)

    return document


def load_model(provider, model, api_key, file_type, file):
    document = load_file(file_type, file)

    document = document.replace("{", "{{").replace("}", "}}")

    system_prompt = f"""Você é um especialista em identificar vieses cognitivos.
Você possui acesso às seguintes informações vindas de um documento {file_type}.
Leia o texto a seguir e responda as perguntas:

####
{document}
####

Perguntas:
- Existe algum viés cognitivo presente? (Sim/Não)
- Quais são esses vieses? (Liste os tipos encontrados)
- Forneça uma breve justificativa para cada viés detectado.
- Estime um percentual geral do nível de viés cognitivo presente no texto (0% sem viés, 100% altamente enviesado).

Utilize as informações fornecidas para basear as suas respostas.
"""

    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("user", "{user_input}"),
        ],
    )

    chat = MODELS_CONFIG[provider]["chat"](model=model, api_key=api_key)
    chat_chain = template | chat

    return chat_chain


def chat_page():
    st.header("🤖 Welcome to the Bias Finder!", divider=True)

    chat_model = st.session_state.get("chat", None)
    chat_memory = st.session_state.get("chat_memory", CHAT_MEMORY)

    if chat_model is None:
        st.error("Please load a file, select a model and start the Bias Finder.")
        st.stop()

    for message in chat_memory.buffer_as_messages:
        chat = st.chat_message(message.type)
        chat.markdown(message.content)

    user_input = st.chat_input("Talk to me...")

    if user_input:
        chat_memory.chat_memory.add_user_message(user_input)
        chat = st.chat_message("human")
        chat.markdown(user_input)

        chat = st.chat_message("ai")
        response = chat.write_stream(
            chat_model.stream(
                {
                    "user_input": user_input,
                    "chat_history": chat_memory.buffer_as_messages,
                }
            )
        )

        chat_memory.chat_memory.add_ai_message(response)
        st.session_state["chat_memory"] = chat_memory


def sidebar():
    tabs = st.tabs(["File Upload", "Model Selection"])

    with tabs[0]:
        file_type = st.selectbox("Select file type", VALID_FILE_TYPES)

        if file_type == "PDF":
            file = st.file_uploader("Upload a PDF file", type="pdf")

        elif file_type == "TXT":
            file = st.file_uploader("Upload a TXT file", type="txt")

    with tabs[1]:
        provider = st.selectbox("Select the model provider", MODELS_CONFIG.keys())
        model = st.selectbox("Select the model", MODELS_CONFIG[provider]["models"])
        api_key = st.text_input(
            f"Enter the API key for the {provider}",
            value=st.session_state.get(f"api_key_{provider}", ""),
        )

        st.session_state[f"api_key_{provider}"] = api_key

    if st.button("Start Bias Finder", use_container_width=True):
        chat = load_model(provider, model, api_key, file_type, file)
        st.session_state["chat"] = chat
        st.session_state["chat_memory"] = ConversationBufferMemory()

        # Mensagem automática do usuário ao iniciar
        initial_message = "Me forneça o relatório de viés cognitivo do texto."
        st.session_state["chat_memory"].chat_memory.add_user_message(initial_message)

        # Obtendo resposta imediata do modelo
        response = chat.invoke(
            {
                "user_input": initial_message,
                "chat_history": [],
            }
        ).content

        st.session_state["chat_memory"].chat_memory.add_ai_message(response)
        st.rerun()


def main():
    with st.sidebar:
        sidebar()
    chat_page()


if __name__ == "__main__":
    main()
