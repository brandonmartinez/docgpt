import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from VectorDatabase import VectorDatabase


class LlmChain:
    store = {}

    def __init__(self, database: VectorDatabase):
        # Capture dependencies
        self.db = database

        # Setup the model
        host = os.getenv('OLLAMA_HOST', "localhost:11434")
        base_url = f"http://{host}"
        self.model = ChatOllama(base_url=base_url, model="mistral")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def invoke(self, query: str):
        retriever = self.db.retriever()

        ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, contextualize_q_prompt
        )

        ### Answer question ###
        qa_system_prompt = """You are textual research assistant for question-answering tasks. \
        Use only the following pieces of retrieved context to answer the question. \
        If you don't know the answer from the context, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        invoked_chain = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "test"}}
        )

        answer = invoked_chain["answer"]

        return answer
