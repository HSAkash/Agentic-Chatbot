from langchain_core.tools import tool
from pyprojroot import here
from langchain_chroma import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from accord.utils.common import get_config
from accord.utils.file_loader import extract_text_content
from glob import glob
from accord import logger
import os

class RAGTool:
    """
    This class creates tools dynamically using the RAG model.
    Based on embedding vector databases directory.
    """

    retriever = {}
    tools = []

    def __init__(self) -> None:
        self.retriever = {}
        self.config = get_config(here("configs/tools_config.yaml"))
        self.embedding_model = GPT4AllEmbeddings()
        self.create_retriever()
        self.create_tools()

    def create_tools(self):
        """
        Create tools dynamically using the RAG model.
        based on how many retrievers are available.
        
        """
        # flash the previous tools
        RAGTool.tools = []
        for key in RAGTool.retriever.keys():
            # get tools documentation from the info file
            info_file_path = os.path.join(
                here(),
                self.config.data_embedding.root_dir,
                key,
                self.config.data_embedding.info_doc_name
            )
            if not os.path.exists(info_file_path):
                continue
            tool_doc = extract_text_content(info_file_path)
            if not tool_doc:
                continue
            # read documentation from the info file
            tool_doc = tool_doc[0].page_content.replace('"', '\\"')  # Escape double quotes

            # create the tool function in the local namespace
            tool_name = key.replace(" ", "_").replace("-", "_")
            # create the tool function
            tool_config = f"""
from langchain_core.tools import tool

@tool
def {tool_name}(query: str) -> str:
    \"\"\"{tool_doc}\"\"\"
    relevant_docs = RAGTool.retriever["{key}"].invoke(query)
    return "\\n\\n".join([doc.page_content for doc in relevant_docs])
"""
            local_namespace = {}
            # execute the tool config in the local namespace
            exec(tool_config, globals(), local_namespace)
            # append the tool to the tools list
            RAGTool.tools.append(local_namespace[tool_name])
            logger.info(f"Tool {tool_name} created successfully")


    def create_retriever(self):
        """
        Create retriever dynamically using the RAG model.
        based on how many vector databases are available.
        """
        for db_path in glob(f"{here()}/{self.config.data_embedding.vectordb_dir}/*"):
            tool_name = os.path.basename(db_path)
            # load vector store using the Chroma vector store
            vectordb = Chroma(
                collection_name=self.config.data_embedding.collection_name,
                persist_directory=db_path,
                embedding_function=self.embedding_model
            )
            # create a retriever using the vector store's search functionality
            retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": self.config.data_embedding.k, "score_threshold": 0.4},
            )
            # add the retriever to the retriever dictionary
            RAGTool.retriever[tool_name] = retriever


if __name__ == "__main__":
    RAGTool()
    logger.info(f"name : {RAGTool.tools[0].name}")
    logger.info(f"args : {RAGTool.tools[0].args}")
    logger.info(f"description : {RAGTool.tools[0].description}")

    query = "population of bangladesh"
    response = RAGTool.tools[0](query)  # Call the first dynamically created tool

    logger.info(f"response : {response}")