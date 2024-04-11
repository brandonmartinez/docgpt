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
    collection_name = "docgpt"

    def __init__(self):
        self.database = Chroma(
            embedding_function=FastEmbedEmbeddings(), persist_directory=".chromadb", collection_name=self.collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, length_function=len, is_separator_regex=False)

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
            search_type="mmr"
        )

    def get_collection(self):
        coll = self.database._client.get_collection(self.collection_name)
        data = coll.get()
        return data

    def documents(self):
        data = self.get_collection()

        if (data is None):
            return []

        documents = data['documents']

        if (documents is None or len(documents) == 0):
            return []

        return documents

    def clear(self):
        self.database._client.delete_collection(self.collection_name)
        self.database.persist()
