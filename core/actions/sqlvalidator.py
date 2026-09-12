import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.logger import get_logger, log_performance

logger = get_logger(__name__)

class SQLValidator(object):

    def __init__(self, llm_client, system_prompt_path: str = "prompts/sql_validator.txt", data_dictionary_folder: str = "") -> None:

        self.system_prompt_path = Path(system_prompt_path)
        self.data_dictionary_folder = data_dictionary_folder
        self.llm_client = llm_client
        self.logger = get_logger(__name__)
        self.prompt_template = self._load_prompt_template()
        self.system_prompt = self._load_system_prompt()
        self.logger.info("SQLValidator has been initialized")
        
    def _load_prompt_template(self) -> str:

        try:
            return self.system_prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"System prompt file not found at: {self.system_prompt_path}") from e
        except IOError as e:
            raise IOError(f"Error reading system prompt file: {e}") from e
       
    def _load_data_dictionary(self, table_name: str) -> Dict[str, str]:

        try:
            file_path = os.path.join(self.data_dictionary_folder, f"{table_name}.json")
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data dictionary not found for table: {table_name}") from e
        except json.JSONDecodeError as e:
            raise IOError(f"Invalid JSON in data dictionary for {table_name}: {e}") from e      
        
    def _format_prompt(self, user_prompt: str, sql_queries: Dict[str, str],
                       table_name: str, prompt_history: Optional[pd.DataFrame] = None
                       ) -> str:
        
        previous_prompts = []
        if prompt_history is not None:
            if not prompt_history.empty and 'prompt_text' in prompt_history.columns:
                previous_prompts = prompt_history['prompt_text'].dropna().tolist()
        self.logger.info("Chat history processed")
                
        data_dictionary = self._load_data_dictionary(table_name)
        self.logger.info(f"Data dict loaded for: {table_name}")
        
                 
        return self.system_prompt.format(user_prompt=user_prompt,                         
                                         sql_queries=json.dumps(sql_queries, indent=2),
                                         data_dictionary=json.dumps(data_dictionary, indent=2),
                                         table_name=table_name,
                                         prompt_history=previous_prompts)
    @staticmethod
    def _extract_json_from_response(
        response_text: str
    ) -> Optional[Dict[str, Any]]:

        try:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_obj = json.loads(json_str)
                return json_obj
            else:
                return None

        except json.JSONDecodeError:
            return None    
        
    def validate_table_queries(self, user_prompt: str, sql_queries: Dict[str, str],
                            table_name: str, prompt_history: Optional[pd.DataFrame] = None
                            ) -> Dict:
  
        try:
            data_dictionary = self._load_data_dictionary(table_name)
            table_description = json.dumps(data_dictionary, indent=2)
            for title, sql_query in sql_queries.items():
                prompt = self.prompt_template.format(
                    user_query=user_prompt,
                    sql_query=sql_query,
                    table_description=table_description
                )
                response = self.llm_client.send_sync_request(prompt)
                llm_response = response["choices"][0]["message"]["content"]
                result = SQLValidator._extract_json_from_response(llm_response)
                if result and all(isinstance(v, str) for v in result.values()):
                    sql_queries[title] = list(result.values())[0]
            return sql_queries
        except Exception as e:
            self.logger.info(f"Error in query validation process - unchanged queries were returned: {e}")
            return sql_queries
        
    @log_performance
    def validate_sql_queries(self, user_prompt: str, sql_queries: dict, prompt_history=None) -> dict:
        # For testing, just return the input dictionary unchanged
        return sql_queries

    def _load_system_prompt(self):
        # Minimal stub for compatibility
        return ""