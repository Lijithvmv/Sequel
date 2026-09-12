import json
from typing import Any, Dict

import pandas as pd

from core.config import AGENT_NAME
from core.logger import get_logger

logger = get_logger(__name__)

def csv_to_dict(db_data: Dict[str, Any], logger) -> Dict[str, str]:
    """
    Convert session CSV data to dictionary format.
    In production, this would retrieve actual CSV data from the session.
    For now, returns empty dict as placeholder.
    """
    try:
        session_id = db_data.get('session_id', '')
        # TODO: Implement actual CSV data retrieval logic
        # This would typically read from session folder or database
        logger.debug(f"Retrieved CSV data for session {session_id}")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving CSV data: {e}")
        return {}

def read_top_chats(db_data: Dict[str, Any], session_folder_path: str, logger) -> pd.DataFrame:
    """
    Read top/recent chats from session data.
    Returns DataFrame with chat history.
    """
    try:
        session_id = db_data.get('session_id', '')
        # Use the new chat history manager for consistency
        from core.actions.chathistory import get_chat_history_manager
        
        chat_manager = get_chat_history_manager(base_path=session_folder_path)
        recent_chats = chat_manager.get_chat_history(session_id, limit=10)
        
        if not recent_chats:
            return pd.DataFrame(columns=['prompt_text', 'response_text'])
        
        # Convert to expected DataFrame format
        df_data = []
        for chat in recent_chats:
            df_data.append({
                'prompt_text': chat.get('user_query', ''),
                'response_text': chat.get('query_result_summary', ''),
                'timestamp': chat.get('timestamp', ''),
                'tables_used': json.dumps(chat.get('tables_referenced', [])),
                'sql_generated': chat.get('generated_sql', '')
            })
        
        df = pd.DataFrame(df_data)
        logger.info(f"Retrieved {len(df)} chat entries for session {session_id}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading chat history: {e}")
        return pd.DataFrame(columns=['prompt_text', 'response_text'])

def answer_from_existing_data(db_data: Dict[str, Any], csv_data_dict: Dict[str, str], 
                            top_chats: pd.DataFrame, llm_client=None) -> Dict[str, Any]:
    """
    Use TaskEvaluator to determine if the current query can be answered from existing session data.
    
    Args:
        db_data: Database data including session_id and user_query
        csv_data_dict: Available CSV data from the session
        top_chats: Previous chat history
        llm_client: LLM client for task evaluation
        
    Returns:
        Dict with CanIAnswerThePrompt flag and response data
    """
    try:
        user_query = db_data.get('user_query', '')
        session_id = db_data.get('session_id', '')
        
        if not user_query:
            logger.warning("No user query provided for existing data check")
            return _create_negative_response("No query provided")
        
        # If no existing data, can't answer from existing data
        if csv_data_dict is None:
            csv_data_dict = {}
        if top_chats is None or top_chats.empty:
            logger.info(f"No existing data for session {session_id}, proceeding with fresh query")
            return _create_negative_response("No existing session data available")
        
        # Prepare data sources for TaskEvaluator
        data_sources = _prepare_data_sources(csv_data_dict, top_chats)
        
        if not data_sources:
            logger.info("No meaningful data sources found in session")
            return _create_negative_response("No meaningful data sources in session")
        
        # Use TaskEvaluator if LLM client is available
        if llm_client:
            try:
                from core.actions.taskevaluator import TaskEvaluator
                
                task_evaluator = TaskEvaluator(llm_client)
                evaluation_result = task_evaluator.evaluate_task(
                    user_prompt=user_query,
                    data_sources=data_sources,
                    prompt_history=top_chats
                )
                
                logger.info(f"TaskEvaluator result for session {session_id}: {evaluation_result}")
                
                # If task is feasible, try to generate answer from existing data
                if evaluation_result.get("feasible", False):
                    return _generate_answer_from_existing_data(
                        user_query, data_sources, top_chats, evaluation_result
                    )
                else:
                    logger.info("TaskEvaluator determined query cannot be answered from existing data")
                    return _create_negative_response("Query requires fresh database access")
                    
            except Exception as e:
                logger.error(f"TaskEvaluator failed: {e}")
                # Fallback to simple heuristic check
                return _simple_existing_data_check(user_query, top_chats)
        else:
            # Fallback to simple heuristic check when no LLM client
            logger.info("No LLM client available, using simple heuristic check")
            return _simple_existing_data_check(user_query, top_chats)
            
    except Exception as e:
        logger.error(f"Error in answer_from_existing_data: {e}")
        return _create_negative_response(f"Error checking existing data: {str(e)}")

def _prepare_data_sources(csv_data_dict: Dict[str, str], top_chats: pd.DataFrame) -> Dict[str, str]:
    """
    Prepare data sources description for TaskEvaluator.
    """
    data_sources = {}
    
    # Add CSV data sources
    for key, data in csv_data_dict.items():
        data_sources[f"csv_data_{key}"] = f"CSV data: {str(data)[:100]}..."
    
    # Add information about previous queries and their results
    if not top_chats.empty:
        recent_queries = []
        for _, row in top_chats.head(5).iterrows():
            query_info = {
                "query": row.get('prompt_text', ''),
                "result": row.get('response_text', ''),
                "tables": row.get('tables_used', '[]')
            }
            if query_info["query"]:
                recent_queries.append(query_info)
        
        if recent_queries:
            data_sources["previous_queries"] = f"Previous queries and results: {json.dumps(recent_queries)}"
    
    return data_sources

def _generate_answer_from_existing_data(user_query: str, data_sources: Dict[str, str], 
                                      top_chats: pd.DataFrame, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an answer using existing session data when TaskEvaluator determines it's feasible.
    """
    try:
        # Look for similar previous queries
        if not top_chats.empty:
            # Simple similarity check - in production, this could be more sophisticated
            user_query_lower = user_query.lower()
            
            for _, row in top_chats.head(10).iterrows():
                prev_query = row.get('prompt_text', '').lower()
                prev_result = row.get('response_text', '')
                
                # Simple keyword overlap check
                user_words = set(user_query_lower.split())
                prev_words = set(prev_query.split())
                overlap = len(user_words.intersection(prev_words))
                
                # If significant overlap and we have a previous result
                if overlap >= 2 and prev_result and len(prev_result) > 10:
                    logger.info(f"Found similar previous query with {overlap} word overlap")
                    
                    return {
                        "CanIAnswerThePrompt": True,
                        "agent_name": AGENT_NAME,
                        "answer_text": f"Based on previous query results: {prev_result}",
                        "generated_sql": row.get('sql_generated', ''),
                        "debug_info": {
                            "source": "existing_session_data",
                            "similar_query": prev_query,
                            "word_overlap": overlap,
                            "evaluation": evaluation_result
                        },
                        "response_metadata": {
                            "answer_img": [],
                            "answer_table": _parse_table_from_result(prev_result),
                            "explanation": "Answer derived from similar previous query in this session"
                        }
                    }
        
        # If no similar query found, still return positive response but indicate need for clarification
        return {
            "CanIAnswerThePrompt": True,
            "agent_name": AGENT_NAME,
            "answer_text": "I found relevant data in your session history, but need to process a fresh query to give you the most accurate answer.",
            "generated_sql": "",
            "debug_info": {
                "source": "existing_session_data_partial",
                "evaluation": evaluation_result,
                "data_sources_count": len(data_sources)
            },
            "response_metadata": {
                "answer_img": [],
                "answer_table": [],
                "explanation": "TaskEvaluator determined query is feasible with existing data"
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating answer from existing data: {e}")
        return _create_negative_response(f"Error processing existing data: {str(e)}")

def _simple_existing_data_check(user_query: str, top_chats: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple heuristic check when TaskEvaluator is not available.
    """
    if top_chats.empty:
        return _create_negative_response("No previous queries in session")
    
    user_query_lower = user_query.lower()
    
    # Check for exact or very similar queries
    for _, row in top_chats.head(5).iterrows():
        prev_query = row.get('prompt_text', '').lower()
        if prev_query and len(prev_query) > 5:
            # Simple similarity check
            if user_query_lower == prev_query or user_query_lower in prev_query:
                prev_result = row.get('response_text', '')
                if prev_result:
                    logger.info("Found exact/similar query match in session history")
                    return {
                        "CanIAnswerThePrompt": True,
                        "agent_name": AGENT_NAME,
                        "answer_text": f"From previous query: {prev_result}",
                        "generated_sql": row.get('sql_generated', ''),
                        "debug_info": {"source": "simple_heuristic_match"},
                        "response_metadata": {
                            "answer_img": [],
                            "answer_table": _parse_table_from_result(prev_result),
                            "explanation": "Answer from identical previous query"
                        }
                    }
    
    return _create_negative_response("No similar queries found in session history")

def _parse_table_from_result(result_text: str) -> list:
    """
    Try to extract table data from result text.
    This is a simple implementation - in production, you'd store structured data.
    """
    try:
        # Simple pattern matching for "X rows returned"
        if "rows returned" in result_text or "rows" in result_text:
            # This is a placeholder - in real implementation, you'd have structured data
            return [["Result"], [result_text]]
        return []
    except Exception:
        return []

def _create_negative_response(reason: str) -> Dict[str, Any]:
    """
    Create a standardized negative response.
    """
    return {
        "CanIAnswerThePrompt": False,
        "agent_name": AGENT_NAME,
        "answer_text": "",
        "response_metadata": {
            "answer_img": [],
            "answer_table": [],
            "explanation": ""
        },
        "debug_info": {
            "reason": reason,
            "source": "answer_from_existing_data"
        }
    }

# process_user_query moved to core.actions.userqueryinterface for better organization 