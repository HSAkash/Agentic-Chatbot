import os
from accord.utils.common import get_config
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentType
from pyprojroot import here
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SQLAgentClass(metaclass=SingletonMeta):
    """
    A specialized SQL agent that interacts with the  SQL database using an LLM (Large Language Model).

    Attributes:
        sql_agent_llm (google gemini): The language model used for interpreting and interacting with the database.
        db (SQLDatabase): The SQL database object, representing the Chinook database.
        sql_agent (Runnable): An agent of operations that maps user questions to SQL tables and executes queries.

    Methods:
        __init__: Initializes the agent by setting up the LLM, connecting to the SQL database, and creating query chains.

    
    """

    def __init__(self) -> None:
        """Initializes the ChinookSQLAgent with the LLM and database connection.
        The agent is created with a GoogleGenerativeAI model and a SQLDatabase object."""
        
        self.config = get_config(here('configs/tools_config.yaml'))
        self.llm = GoogleGenerativeAI(model="gemini-pro")

        self.db = SQLDatabase.from_uri(f"sqlite:///{here(self.config.sql_agent.db_path)}")
        print(self.db.get_usable_table_names())
        self.agent_executor = create_sql_agent(
            self.llm,
            db = self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose = False,
            handle_parsing_errors=True
        )


@tool
def query_sqldb(query: str) -> str:
    """Query the  SQL Database. Input should be a search query."""
    # Create an instance of ChinookSQLAgent
    sql_agent = SQLAgentClass()
    response = {}
    try:
        response = sql_agent.agent_executor.invoke(query)
    except Exception as e:
        pass

    return response.get("output", "I don't know?")


if __name__ == "__main__":
    print(query_sqldb("What are all the genres of Alanis Morissette songs?"))