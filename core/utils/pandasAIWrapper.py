# General
from typing import Any, List, Optional

import pandas as pd

# Third Party
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult
from pandasai import SmartDataframe, SmartDatalake
from pydantic import Field

from core.logger import get_logger, log_performance

logger = get_logger(__name__)

class CustomLLMWrapper(BaseChatModel):
    """
    Custom LLM wrapper to make our LLMClient compatible with Langchain.
    """        
    client: Any = Field(..., description="LLM client instance")
    config: Any = Field(..., description="Configuration object")

    class Config(object):
        """
        Configuration for this pydantic object.
        """
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """ Return type of LLM."""
        return "custom_llm"
    
    @log_performance
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate LLM results with chat-style formatting.
        
        Args:
            messages (List[BaseMessage]): The conversation messages.
            prompt (str): The prompt to send to the LLM
            run_manager (Optional[CallbackManagerforLLMRun]): Callback manager
            **kwargs: Additional keyword arguments
        
        Returns:
            List[Generated]: List of chat-formatted generations
        """ 
        logger.info("Starting PandasAI Wrapper LLM Calls")

        prompt = " ".join([m.content for m in messages])
        
        try:
            response = self.client.send_sync_request(prompt)
            message_content = response["choices"][0]["message"]["content"]
            
            chat_generation = ChatGeneration(
                text=message_content,
                generation_info={},
                message=AIMessage(content=message_content)
            )

            return ChatResult(generations=[chat_generation])
            
        except KeyError as e:
            raise ValueError(f"Unexpected response structure: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error calling LLM: {str(e)}")


class DataAnalyzer(object):
    """
    Data analyzer using Langchain's PandasAgent with custom LLM.
    """        
    def __init__(self, llm_client: Any, config: Any):
        """
        Initialize the data analyzer.
        
        Args:
            llm_client: Intance of LLMClient
            config: Configuration object        
        """
        self.llm = CustomLLMWrapper(client=llm_client, config=config)
        
    def create_agent(
        self, 
        dataframes: dict[str, pd.DataFrame],
        save_path: str) -> Any:
        """
        Create a PandasAgent for one or more DataFrames.
        
        Args:
            dataframes: Dictionary with DataFrame names (keys) and DataFrames as values.
            save_path: The path where the generated charts should be saved.
        
        Returns:
            Any: Configured PandasAgent instance    
        """

        logger.info("Starting Agent Orchestrator Execution")
        
        smart_dfs: List[SmartDataframe] = []
        
        original_names = list(dataframes.keys())
        
        for name, df in dataframes.items():
            smart_df = SmartDataframe(df, name=name, config={"llm": self.llm,
                                               "enable_cache": False,
                                               "save_logs": False
                                               })
            smart_dfs.append(smart_df)
        
        pandas_ai = SmartDatalake(smart_dfs, config={"llm": self.llm,
                                               "save_charts": True,
                                               "save_charts_path": save_path,
                                               "enable_cache": False,
                                               "save_logs": False
                                               })
        
        for i, name in enumerate(original_names):
            if i < len(pandas_ai.dfs):
                pandas_ai.dfs[i].name = name
        
        return pandas_ai

    @staticmethod      
    def analyze(agent: Any, query: str) -> str:
        """
        Analyze the  DataFrame using the agent.
        
        Args:
            agent: The PandasAgent instance
            query (str): The analysis query
        
        Returns:
            str: Analysis results
        """ 

        logger.info("Starting PandasAI Analyzer")

        try:
            return agent.chat(query)
        except Exception as e:
            raise ValueError(f"Error during analysis: {str(e)}")  