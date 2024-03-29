from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from VectorDatabase import VectorDatabase

class LlmChain:
    def __init__(self,database: VectorDatabase):
        self.model = ChatOllama(model="mistral")
        self.db = database

    def invoke(self, prompt:str, query: str):
        retriever = self.db.retriever()

        chain = ({"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser())

        return chain.invoke(query)