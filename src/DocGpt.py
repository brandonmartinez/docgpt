from langchain.prompts import PromptTemplate

from LlmChain import LlmChain

class DocGPT:
    db = None
    chain = None

    def __init__(self, llmchain: LlmChain):
        # Setup dependencies
        self.chain = llmchain

        # Configure defaults
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s>
            <context>
            {context}
            </context>

            Question: {question}
            Answer:
            """
        )

    def ask(self, query: str) -> str:
        output = self.chain.invoke(self.prompt, query)

        return output

    def clear(self):
        print("Clearing")
        # self.db = None
        # self.chain = None