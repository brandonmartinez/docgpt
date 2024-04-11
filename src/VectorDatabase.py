from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata

# Docs for Chroma: https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html


class VectorDatabase:
    database = None
    text_splitter = None

    def __init__(self):
        self.database = Chroma(
            embedding_function=FastEmbedEmbeddings(), persist_directory=".chromadb")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100)

    def ingest(self, file_path: str, file_extension: str):
        if file_extension == '.pdf':
            docs = PyPDFLoader(file_path=file_path).load()
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
            documents = loader.load()
            docs = self.text_splitter.split_documents(documents)
        else:
            raise ValueError("Unsupported file type %s from %s. Please provide a .pdf or .txt file." % (
                file_extension, file_path))

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.database.add_documents(documents=chunks)
        self.database.persist()

    def retriever(self):
        return self.database.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
