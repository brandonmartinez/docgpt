from DocGpt import DocGPT
from VectorDatabase import VectorDatabase
from LlmChain import LlmChain
from streamlit_chat import message
import os
import streamlit as st
import tempfile

database = VectorDatabase()
llmchain = LlmChain(database)

st.set_page_config(page_title="DocGPT")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ''


def read_and_save_file():
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            file_extension = os.path.splitext(file.name)[1]
            database.ingest(file_path, file_extension)
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = DocGPT(llmchain)

    st.header("DocGPT")

    st.sidebar.header("Vector Data")
    st.sidebar.subheader("Add file to vector database")
    st.sidebar.file_uploader(
        "Upload document",
        type=["pdf", "txt"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.sidebar.subheader("Data currently in vector database")
    # st.sidebar.write(database.documents())
    st.sidebar.dataframe(database.documents(), use_container_width=True)
    st.sidebar.button("Clear database", on_click=database.clear)

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
