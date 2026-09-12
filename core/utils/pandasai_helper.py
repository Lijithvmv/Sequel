"""
PandasAI v3 helper module for enhanced data analysis and follow-up question handling.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .query_context_manager import QueryContext, get_query_context_manager
from .simple_data_analyzer import get_simple_analyzer

logger = logging.getLogger(__name__)

class PandasAIHelper:
    """Helper class for PandasAI v3 integration using semantic dataframes"""
    
    def __init__(self):
        self.available = False
        self.semantic_dataframes = {}
        self.pai = None
        self.fallback_analyzer = get_simple_analyzer()
        self.context_manager = get_query_context_manager()
        self._check_availability()
    
    def _check_availability(self):
        """Check if PandasAI is available and initialize"""
        try:
            import pandasai as pai
            self.pai = pai
            self.available = True
            self._configure_llm()
            logger.info("PandasAI is available and initialized")
        except ImportError as e:
            self.available = False
            logger.warning(f"PandasAI not available: {e}. Using fallback analyzer only.")
    
    def _configure_llm(self):
        """Configure PandasAI v3 with existing LLM client"""
        try:
            from core.connectors.LLMClient import LLMClient
            
            # Create a wrapper to make existing LLM compatible with PandasAI v3
            class LLMWrapper:
                def __init__(self, llm_client):
                    self.client = llm_client
                
                def call(self, prompt):
                    response = self.client.send_sync_request(prompt)
                    return response["choices"][0]["message"]["content"]
            
            llm_wrapper = LLMWrapper(LLMClient())
            
            # Configure PandasAI v3 with the LLM
            self.pai.config.set({"llm": llm_wrapper})
            logger.info("Configured PandasAI v3 with existing LLM client")
        except Exception as e:
            logger.error(f"Error configuring LLM for PandasAI v3: {e}")
            self.available = False
    
    def create_smart_dataframe(self, df: pd.DataFrame, session_id: str, query_context: QueryContext = None) -> bool:
        """Create and store a semantic dataframe for analysis using PandasAI v3 with enhanced context"""
        try:
            # Always store in fallback analyzer with enhanced context
            context_str = query_context.original_query if query_context else ""
            fallback_success = self.fallback_analyzer.store_dataframe(df, session_id, context_str)
            
            # Store the query context for intelligent follow-up detection
            if query_context:
                self.context_manager.store_context(query_context)
            
            # Create semantic dataframe if PandasAI v3 is available
            if self.available and self.pai is not None:
                # Convert to PandasAI v3 semantic dataframe
                semantic_df = self.pai.DataFrame(df)
                
                self.semantic_dataframes[session_id] = {
                    "semantic_df": semantic_df,
                    "original_df": df,
                    "created_at": datetime.now(),
                    "query_context": query_context,
                    "context": context_str
                }
                logger.info(f"Created semantic dataframe for session {session_id} with {len(df)} rows")
                return True
            else:
                logger.info(f"Using fallback analyzer for session {session_id} with {len(df)} rows")
                return fallback_success
            
        except Exception as e:
            logger.error(f"Error creating semantic dataframe: {e}")
            # Try fallback even if PandasAI fails
            return self.fallback_analyzer.store_dataframe(df, session_id, context_str)
    
    def query_dataframe(self, session_id: str, query: str) -> Tuple[Optional[Any], Optional[str]]:
        """Query the semantic dataframe using PandasAI v3 or fallback analyzer"""
        
        # Try PandasAI v3 first if available
        if self.available and session_id in self.semantic_dataframes:
            try:
                session_data = self.semantic_dataframes[session_id]
                semantic_df = session_data["semantic_df"]
                
                # Use PandasAI v3 chat API with semantic dataframe
                result = self.pai.chat(query, semantic_df)
                
                logger.info(f"Successfully processed PandasAI v3 query for session {session_id}")
                return result, None
                
            except Exception as e:
                logger.warning(f"PandasAI v3 query failed, falling back to simple analyzer: {e}")
        
        # Use fallback analyzer
        if self.fallback_analyzer.can_answer_query(query, session_id):
            result, error = self.fallback_analyzer.analyze_query(query, session_id)
            if result is not None:
                logger.info(f"Successfully processed query with fallback analyzer for session {session_id}")
                return result, None
            else:
                return None, error
        
        return None, "Query cannot be processed with available analyzers"
    
    def can_answer_query(self, query: str, session_id: str) -> Tuple[bool, Optional[QueryContext], float]:
        """
        Enhanced follow-up detection using intelligent context analysis.
        
        Returns:
            (can_answer, relevant_context, confidence_score)
        """
        # Use the context manager for intelligent follow-up detection
        is_followup, relevant_context, confidence_score = self.context_manager.is_followup_query(query, session_id)
        
        if is_followup and relevant_context:
            # Additional check: do we have the actual dataframe available?
            has_dataframe = False
            
            # Check PandasAI storage
            if self.available and session_id in self.semantic_dataframes:
                has_dataframe = True
            
            # Check fallback analyzer storage
            elif self.fallback_analyzer.can_answer_query(query, session_id):
                has_dataframe = True
            
            return has_dataframe, relevant_context, confidence_score
        
        return False, None, 0.0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about the dataframe for a session"""
        # Try PandasAI first
        if session_id in self.semantic_dataframes:
            data = self.semantic_dataframes[session_id]
            df = data["original_df"]
            
            return {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "created_at": data["created_at"].isoformat(),
                "context": data["context"],
                "source": "pandasai"
            }
        
        # Fallback to simple analyzer
        info = self.fallback_analyzer.get_session_info(session_id)
        if info:
            info["source"] = "fallback"
        return info
    
    def clear_session(self, session_id: str) -> bool:
        """Clear the dataframe for a session"""
        success = False
        
        if session_id in self.semantic_dataframes:
            del self.semantic_dataframes[session_id]
            logger.info(f"Cleared PandasAI data for session {session_id}")
            success = True
        
        # Also clear from fallback
        fallback_success = self.fallback_analyzer.clear_session(session_id)
        
        return success or fallback_success
    
    def get_available_sessions(self) -> List[str]:
        """Get list of sessions with available dataframes"""
        pandas_sessions = set(self.semantic_dataframes.keys())
        fallback_sessions = set(self.fallback_analyzer.get_available_sessions())
        return list(pandas_sessions.union(fallback_sessions))
    
    def get_query_context(self, session_id: str) -> Optional[QueryContext]:
        """Get the most recent query context for a session"""
        recent_contexts = self.context_manager.get_recent_context(session_id, limit=1)
        return recent_contexts[0] if recent_contexts else None

# Global instance
pandasai_helper = PandasAIHelper()

def get_pandasai_helper() -> PandasAIHelper:
    """Get the global PandasAI helper instance"""
    return pandasai_helper