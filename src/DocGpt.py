from langchain.prompts import PromptTemplate

from LlmChain import LlmChain


class DocGPT:
    db = None
    chain = None

    def __init__(self, llmchain: LlmChain):
        # Setup dependencies
        self.chain = llmchain

        promptText = """
        ### [INST]
        Instruction: You are a textual research assistant for question-answering tasks over a set of documents.
        If you don't know the answer, just say that you don't know. Don't mention the documents, just the contents within.
        Use the following context to answer the question:

        {context}

        ### QUESTION:
        {question}

        [/INST]
        """

        # Configure defaults
        self.prompt = PromptTemplate.from_template(promptText)

    def ask(self, query: str) -> str:
        output = self.chain.invoke(self.prompt, query)

        return output
