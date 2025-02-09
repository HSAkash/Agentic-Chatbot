import os
from tqdm import tqdm
from glob import glob
from accord import logger
from accord.utils.common import (get_config, create_directory)
from accord.utils.file_loader import load_directory_files
from langchain_chroma import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PrepareVectorDB:
    """
    A class to prepare and manage a Vector Database (VectorDB) using documents from a specified directory.
    The class performs the following tasks:
    - Loads and splits documents (PDFs).
    - Splits the text into chunks based on the specified chunk size and overlap.
    - Embeds the document chunks using a specified embedding model.
    - Stores the embedded vectors in a persistent VectorDB directory.

    Attributes:
        doc_dir (str): Path to the directory containing documents (PDFs) to be processed.
        chunk_size (int): The maximum size of each chunk (in characters) into which the document text will be split.
        chunk_overlap (int): The number of overlapping characters between consecutive chunks.
        embedding_model (str): The name of the embedding model to be used for generating vector representations of text.
        vectordb_dir (str): Directory where the resulting vector database will be stored.
        collection_name (str): The name of the collection to be used within the vector database.

    Methods:
        path_maker(file_name: str, doc_dir: str) -> str:
            Creates a full file path by joining the given directory and file name.

        run() -> None:
            Executes the process of reading documents, splitting text, embedding them into vectors, and 
            saving the resulting vector database. If the vector database directory already exists, it skips
            the creation process.
    """

    def __init__(
        self,
        doc_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        vectordb_dir: str,
        collection_name: str
    ) -> None:

        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = GPT4AllEmbeddings()
        self.vectordb_dir = os.path.join(vectordb_dir, os.path.basename(doc_dir))
        self.collection_name = collection_name
        self.batch_size = batch_size

    
    def create_vector_store(self, documents) -> None:
        """
        Create a vector store from a text file and save it to disk.

        Args:
            documents (list): The list of documents.
        """

        
        # Create a vector store using the Chroma vector store
        current_doc_count = 0
        if os.path.exists(self.vectordb_dir):
            db = Chroma(
                persist_directory=self.vectordb_dir,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            current_doc_count = db._collection.count()
        else:
            db = Chroma.from_documents(
                documents[current_doc_count:self.batch_size],
                embedding_function=self.embedding_model,
                persist_directory=self.vectordb_dir,
                collection_name=self.collection_name
            )
            
        for i in tqdm(range(current_doc_count, len(documents), self.batch_size)):
            db.add_documents(documents[i:i+self.batch_size])



    def run(self):
        """
        Executes the main logic to create and store document embeddings in a VectorDB.

        If the vector database directory doesn't exist:
        - It loads PDF documents from the `doc_dir`, splits them into chunks,
        - Embeds the document chunks using the specified embedding model,
        - Stores the embeddings in a persistent VectorDB directory.

        If the directory already exists, it skips the embedding creation process.

        Prints the creation status and the number of vectors in the vector database.

        Returns:
            None
        """
        logger.info(f"Creating VectorDB for documents in '{self.doc_dir}'")

        create_directory(self.vectordb_dir)

        docs_list = load_directory_files(self.doc_dir)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)
        # Add to vectorDB
        self.create_vector_store(doc_splits)

        logger.info(f"VectorDB created for '{self.doc_dir}' with {len(doc_splits)} vectors")


if __name__ == "__main__":

    app_config = get_config("configs/tools_config.yaml")

    unstructured_docs_dirs = glob(app_config.data_embedding.data_dir_re)
    for doc_dir in unstructured_docs_dirs:
        # Prepare the VectorDB for each unstructured docs directory
        chunk_size = app_config.data_embedding.chunk_size
        chunk_overlap = app_config.data_embedding.chunk_overlap
        vectordb_dir = app_config.data_embedding.vectordb_dir
        collection_name = app_config.data_embedding.collection_name
        batch_size = app_config.data_embedding.batch_size

        prepare_db_instance = PrepareVectorDB(
            doc_dir=doc_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            vectordb_dir=vectordb_dir,
            collection_name=collection_name)

        prepare_db_instance.run()

