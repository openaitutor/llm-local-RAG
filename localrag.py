from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document, Settings
from llama_index.llms.ollama import Ollama
import docx2txt

llm = Ollama(model="gemma", request_timeout=600.0)

from langchain_community.embeddings import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

import streamlit as st

st.set_page_config(page_title="local llm RAG webapp", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("local llm RAG")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "please ask"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        Settings.llm = llm
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_documents(docs)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)