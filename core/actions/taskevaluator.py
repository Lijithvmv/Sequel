import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from core.logger import get_logger

# Set Logger
logger = get_logger(__name__)

class TaskEvaluator(object):

    def __init__(self, llm_client, system_prompt_path: Union[str, Path] = "prompts/task_evaluator.txt") -> None:

        self.system_prompt_path = Path(system_prompt_path)
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()
        self.system_prompt = self._load_system_prompt()
        self.logger = get_logger(__name__)
        
        self.logger.info("TaskEvaluator has been initialized")
        
    def _load_prompt_template(self) -> str:

        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"System prompt file not found at: {self.system_prompt_path}") from e
        except IOError as e:
            raise IOError(f"Error reading system prompt file: {e}") from e
        
    def _load_system_prompt(self) -> str:

        try:
            return self.system_prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"System prompt file not found at: {self.system_prompt_path}") from e
        except IOError as e:
            raise IOError(f"Error reading system prompt file: {e}") from e
        
    def _format_prompt(self, user_prompt: str, data_sources: Dict[str, str],
                       prompt_history: Optional[pd.DataFrame] = None) -> str:
        
        if prompt_history is not None:
            if not prompt_history.empty and 'prompt_text' in prompt_history.columns:
                prompt_history['prompt_text'].dropna().tolist()
                
        prompt = self.prompt_template.format(
            user_query=user_prompt,
            data_sources=json.dumps(list(data_sources.keys()))
        )
        return prompt
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
        
    def evaluate_task(self, user_prompt: str, data_sources: Dict[str, str],
                      prompt_history: Optional[pd.DataFrame] = None) -> Dict[str, bool]:

        result = {"feasible": False}
        
        if data_sources:       
            try:
                self.logger.info(f"Data Sources provided to the TaskEvaluator: {data_sources}")
                formatted_prompt = self._format_prompt(user_prompt, data_sources, prompt_history)
                response = self.llm_client.send_sync_request(formatted_prompt)
                llm_response = response["choices"][0]["message"]["content"]
                parsed_result = TaskEvaluator._extract_json_from_response(llm_response)

                self.logger.info(f"TaskEvaluator has responded with: {parsed_result}")
                
                if parsed_result and "feasible" in parsed_result:
                    result = parsed_result
                
            except Exception as e:
                self.logger.info(f"Error in TaskEvaluator: {e}")
            
        return result    
        

    def select_agent(self, user_prompt: str, data_sources: Dict[str, str],
                     prompt_history: Optional[pd.DataFrame] = None) -> Dict[str, bool]:

        try:
            self.logger.info(f"Data Sources provided to the AgentSelector: {data_sources}")
            formatted_prompt = self._format_prompt(user_prompt, data_sources, prompt_history)
            self.logger.info("Prompt created")
            response = self.llm_client.send_sync_request(formatted_prompt)
            llm_response = response["choices"][0]["message"]["content"]
            result = TaskEvaluator._extract_json_from_response(llm_response)

            self.logger.info(f"AgentSelector has responded with: {result}")
            
            if result and "needs_plotting" in result and "needs_analytics" in result:
                return result
            return {"needs_plotting": True, "needs_analytics": True}
             
        except Exception as e:
            self.logger.info(f"Error in AgentSelector: {e}")
            return {"needs_plotting": True, "needs_analytics": True}