import logging
import os
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.actions.queryexecutor import QueryExecutor
from core.actions.querygenerator import QueryGenerator
from core.actions.resultsynthesizer import (
    add_incomplete_data_warning,
    process_synthesized_results,
    synthesize_results,
)
from core.actions.sqlvalidator import SQLValidator

# Updated import to use TableSelector class
from core.actions.tableselector import TableSelector
from core.actions.userqueryinterface import UserQueryInterface, process_user_query
from core.config import AGENT_NAME
from core.connectors.LLMClient import LLMClient

# Import your existing modules (these would need to be available)
from core.logger import get_logger, set_session_id
from core.utils.data_processing import csv_to_dict, read_top_chats
from core.utils.database import (
    get_filter_value,
    insert_session_chat_data,
    read_session_chat_data,
)
from core.utils.explanation import prompt_explainer_wrapper
from core.utils.followup_detector import get_followup_detector
from core.utils.sqlfilter import add_sql_filter

app = FastAPI(
    title="Data Insight API",
    description="API for processing data insight requests and generating analytics",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add these configuration variables for Hybrid Table Selector
HYBRID_TABLE_SELECTOR_CONFIG = {
    "use_ai": os.getenv("USE_AI_TABLE_SELECTION", "true").lower() == "true",
    "hybrid_mode": os.getenv("TABLE_SELECTOR_MODE", "ai_first"),  # "ai_first", "rule_first", "both"
    "ai_timeout": float(os.getenv("AI_TIMEOUT", "5.0")),
    "fallback_threshold": float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.3"))
}

# Initialize shared clients using Config
query_executor = QueryExecutor()
llm_client = LLMClient()


# Pydantic models for request/response validation
class DataInsightRequest(BaseModel):
    session_id: Optional[str] = None
    user_query: str
    chat_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str


# Removed non-essential Pydantic models for production simplicity


session_folder_path = "./sessions"
os.makedirs(session_folder_path, exist_ok=True)

table_descriptions_file = "table_descriptions.csv"
table_selector_instruction = "table_selector_instruction.txt"
query_generator_prompt = "prompts/query_generator.txt"
sql_validator_prompt = "prompts/sql_validator.txt"
data_dictionary_folder = "data_dictionary"


def log_performance(f):
    """Decorator to log performance metrics"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = f(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Function {f.__name__} completed in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Function {f.__name__} failed after {duration:.2f} seconds: {str(e)}")
            raise

    return decorated_function


@log_performance
def process_data_insight(session_id, user_query, chat_id, llm_client, query_executor):
    """Main data insight processing function with hybrid table selection and PandasAI integration"""

    set_session_id(session_id)
    logger = get_logger(__name__)
    
    # CRITICAL DEBUG: Log every request
    print(f"[DEBUG] PROCESSING REQUEST - Session: {session_id}, Query: '{user_query}'")
    print(f"[DEBUG] Chat ID: {chat_id}")

    # Check for follow-up queries using chat history
    followup_detector = get_followup_detector()
    
    # Get recent chat history for follow-up detection
    try:
        from core.actions.chathistory import get_chat_history_manager
        chat_manager = get_chat_history_manager()
        recent_chats = chat_manager.get_chat_history(session_id, limit=3)
        
        # Detect if this is a follow-up query
        is_followup, previous_context, confidence = followup_detector.detect_followup(user_query, recent_chats)
        
        print(f"[DEBUG] Follow-up analysis: Query='{user_query[:50]}...', Confidence={confidence:.2f}, Is_followup={is_followup}")
        
        if is_followup and previous_context:
            logger.info(f"Processing follow-up query with confidence {confidence:.2f}")
            
            # Get the stored answer_table from previous context
            stored_answer_table = previous_context.get('metadata', {}).get('answer_table')
            if not stored_answer_table:
                # Try to get it from query_result field (if stored differently)
                stored_answer_table = previous_context.get('query_result')
            
            if stored_answer_table:
                print(f"[DEBUG] Found stored answer_table with {len(stored_answer_table)} rows")
                
                # Convert answer_table to DataFrame
                df = followup_detector.convert_answer_table_to_dataframe(stored_answer_table)
                
                if df is not None:
                    print(f"[DEBUG] Converted to DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
                    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
                    
                    # Create context for PandasAI
                    context_description = (f"Previous query: {previous_context.get('user_query', '')}\n"
                                         f"Tables: {', '.join(previous_context.get('tables_referenced', []))}\n"
                                         f"SQL: {previous_context.get('generated_sql', '')}")
                    
                    # Create PandasAI agent (using your exact DataAnalyzer pattern)
                    session_save_path = os.path.join("./sessions", session_id)
                    os.makedirs(session_save_path, exist_ok=True)
                    
                    agent = followup_detector.create_pandasai_agent(df, context_description, session_save_path)
                    
                    if agent:
                        # Query using PandasAI (your exact analyze pattern)
                        result, error = followup_detector.query_pandasai_agent(agent, user_query)
                        
                        if result:
                            print("[DEBUG] PandasAI query successful")
                            logger.info("Successfully answered follow-up query using PandasAI")
                            
                            # Handle different result types
                            if isinstance(result, list) and len(result) > 1 and isinstance(result[0], list):
                                # It's already in answer_table format
                                answer_table = result
                                answer_text = f"Found {len(result)-1} matching records"
                            else:
                                # It's a string or other format
                                answer_table = [["Result"], [str(result)]]
                                answer_text = str(result)
                            
                            return {
                                "CanIAnswerThePrompt": True,
                                "agent_name": AGENT_NAME,
                                "answer_text": answer_text,
                                "response_metadata": {
                                    "answer_img": [],
                                    "answer_table": answer_table,
                                    "explanation": f"Answered as follow-up using PandasAI. Previous query: {previous_context.get('user_query', '')}",
                                    "source": "pandasai_followup",
                                    "generated_sql": previous_context.get('generated_sql', ''),
                                    "follow_up_confidence": confidence
                                }
                            }
                        else:
                            print(f"[DEBUG] PandasAI query failed: {error}")
                            logger.warning(f"PandasAI follow-up query failed: {error}")
                    else:
                        print("[DEBUG] PandasAI not available, falling back to SQL query")
                else:
                    print("[DEBUG] Failed to convert answer_table to DataFrame")
            else:
                print("[DEBUG] No stored answer_table found in previous context")
        else:
            print(f"[DEBUG] Not a follow-up query (confidence: {confidence:.2f})")
            
    except Exception as e:
        print(f"[DEBUG] Error in follow-up detection: {e}")
        logger.error(f"Follow-up detection error: {e}")
        # Continue to normal processing

    # Create session folder if that doesn't exist
    session_path = os.path.join(session_folder_path, f"{session_id}")
    os.makedirs(session_path, exist_ok=True)

    # Check if Session Data exist in DB
    session_df = read_session_chat_data({'session_id': session_id}, logger)

    # Write Chat to DB
    db_data = {
        "session_id": session_id,
        "chat_id": chat_id,
        "source_agent": "data_insight",
        "user_query": user_query
    }

    if session_df.empty:
        top_chats = pd.DataFrame(columns=['prompt_text', 'response_text'])
    else:
        # make dictionary like "title:CSV data"
        csv_to_dict(db_data, logger)
        top_chats = read_top_chats(db_data, session_folder_path, logger)

    insert_session_chat_data(db_data, logger)

    # DISABLED: Legacy follow-up detection replaced by enhanced context-aware logic above
    # The enhanced follow-up detection (lines 130-197) handles this much better
    # response = answer_from_existing_data(db_data, csv_data_dict, top_chats, llm_client)
    # if response["CanIAnswerThePrompt"]:
    #     logger.info(f"Answering query from existing session data: {response.get('debug_info', {}).get('source', 'unknown')}")
    #     return response

    # Use UserQueryInterface to properly merge query with chat history context
    merged_query = process_user_query(session_id, user_query, top_chats)
    logger.info(f"Query processing - Original: '{user_query}', Merged: '{merged_query}'")

    # Check if user query exists
    if not merged_query:
        return {
            "CanIAnswerThePrompt": False,
            "agent_name": AGENT_NAME,
            "answer_text": "User query not found",
            "response_metadata": {
                "answer_img": [],
                "answer_table": [],
                "explanation": ""
            }
        }

    # UPDATED: Use Hybrid TableSelector with enhanced configuration
    try:
        table_selector = TableSelector(
            llm_client=llm_client,
            table_descriptions_file=table_descriptions_file,
            instruction_file=table_selector_instruction,
            **HYBRID_TABLE_SELECTOR_CONFIG  # Pass configuration
        )

        # Enhanced table selection with detailed logging
        start_time = time.time()
        selected_tables = table_selector.run(merged_query)
        selection_time = time.time() - start_time

        # Log selection details for monitoring and debugging
        logger.info(f"Table selection completed in {selection_time:.3f}s. "
                    f"Method: {selected_tables.get('selection_method', 'unknown')}, "
                    f"Strategy: {selected_tables.get('strategy_used', 'unknown')}, "
                    f"Tables found: {selected_tables.get('relevant_tables', [])}")

        # Log AI performance metrics if available
        if 'ai_latency' in selected_tables:
            logger.info(f"AI table selection latency: {selected_tables['ai_latency']:.3f}s")

    except Exception as e:
        logger.error(f"Table selection failed: {e}")
        # Fallback to rule-based only in case of complete failure
        try:
            from core.actions.tableselector import RuleBasedTableSelector
            fallback_selector = RuleBasedTableSelector()
            selected_tables = fallback_selector.run(merged_query)
            selected_tables['selection_method'] = 'emergency_fallback'
            logger.warning("Used emergency rule-based fallback for table selection")
        except Exception as fallback_error:
            logger.error(f"Emergency fallback also failed: {fallback_error}")
            return {
                "CanIAnswerThePrompt": False,
                "agent_name": AGENT_NAME,
                "answer_text": "Technical issue with table selection. Please try again.",
                "response_metadata": {
                    "answer_img": [],
                    "answer_table": [],
                    "explanation": ""
                }
            }

    # Check if relevant tables were found
    if not selected_tables['relevant_tables_found']:
        # Enhanced error message with selection method info
        selection_method = selected_tables.get('selection_method', 'unknown')
        return {
            "CanIAnswerThePrompt": False,
            "agent_name": AGENT_NAME,
            "answer_text": f"Your query couldn't be matched to any available information in our database "
                           f"(searched using {selection_method} method). Please try rephrasing your question "
                           f"or specify which type of data you are looking for.",
            "response_metadata": {
                "answer_img": [],
                "answer_table": [],
                "explanation": "",
                "table_selection_debug": {
                    "method": selection_method,
                    "strategy": selected_tables.get('strategy_used', 'unknown'),
                    "confidence_scores": selected_tables.get('confidence_scores', {}),
                    "reasoning": selected_tables.get('reasoning', 'No reasoning available')
                }
            }
        }

    # Continue with existing query generation logic...
    # Generate SQL queries using QueryGenerator with explicit llm_client
    query_generator = QueryGenerator(
        llm_client=llm_client,
        single_table_descriptions_file="dummy.json",  # Dummy value, not used
        query_generator_prompt=query_generator_prompt
    )

    # Pass enhanced table information to query generator
    relevant_tables = selected_tables['relevant_tables']
    confidence_scores = selected_tables.get('confidence_scores', {})

    # Enhanced debug info with table selection details
    table_selection_debug = {
        "selection_method": selected_tables.get('selection_method', 'unknown'),
        "strategy_used": selected_tables.get('strategy_used', 'unknown'),
        "confidence_scores": confidence_scores,
        "reasoning": selected_tables.get('reasoning', ''),
        "selection_time": selection_time
    }

    sql_queries, query_generator_flag, debug_info = query_generator.generate_sql_queries(
        merged_query,
        relevant_tables,
        error=""
    )

    # Add table selection info to debug_info
    if isinstance(debug_info, dict):
        debug_info['table_selection'] = table_selection_debug
    else:
        debug_info = {'table_selection': table_selection_debug}

    # Check if SQL queries were generated successfully
    if query_generator_flag == 0:
        return {
            "CanIAnswerThePrompt": False,
            "agent_name": AGENT_NAME,
            "answer_text": "I couldn't process your request at this moment. There was a temporary issue with our search system. Please try again the same request in a few moments.",
            "response_metadata": {
                "answer_img": [],
                "answer_table": [],
                "explanation": "",
                "table_selection_debug": table_selection_debug
            }
        }

    # SQL Query Validator
    sql_validator = SQLValidator(
        llm_client=llm_client,
        system_prompt_path=sql_validator_prompt,
        data_dictionary_folder=data_dictionary_folder
    )
    sql_queries = sql_validator.validate_sql_queries(merged_query, sql_queries, top_chats)

    # Add access filter to SQL queries
    filter_value = get_filter_value(merged_query, top_chats)
    sql_queries = add_sql_filter(queries=sql_queries, filter_value=filter_value)

    # Connect to the database before executing queries
    query_executor.connect()
    
    try:
        # Collect all generated SQLs and results
        answer_tables = []
        generated_sqls = []
        answer_texts = []
        for table, queries in sql_queries.items():
            for title, sql in queries.items():
                print(f"FINAL SQL EXECUTED: {sql}")
                df, error = query_executor.execute_single_query(sql)
                print(f"RESULT DATAFRAME (first 5 rows):\n{df.head()}")
                generated_sqls.append(sql)
                if not df.empty:
                    answer_tables.append([df.columns.tolist()] + df.values.tolist())
                    answer_texts.append(f"Results for {table}: {len(df)} rows")
                else:
                    answer_texts.append(f"No results for {table}")
    finally:
        # Always close the connection
        query_executor.close_connection()
    # Use the first non-empty table for display
    answer_table = next((tbl for tbl in answer_tables if len(tbl) > 1), [])
    answer_text = "\n".join(answer_texts)
    generated_sql = "\n".join(generated_sqls)

    # Data storage is now handled by chat history (answer_table stored in metadata)
    print(f"DEBUG: Answer table ready for storage - {len(answer_table) if answer_table else 0} rows")
    print("DEBUG: Will be stored in chat history for follow-up queries")

    # DEBUG: Print what will be sent to the UI
    print("DEBUG: answer_table =", answer_table)
    print("DEBUG: generated_sql =", generated_sql)
    print("DEBUG: answer_text =", answer_text)

    # Store the chat history entry
    try:
        get_chat_history_manager()
        user_interface = UserQueryInterface(llm_client=llm_client, session_id=session_id)
        user_interface.store_query_context(
            session_id=session_id,
            user_query=user_query,
            tables_referenced=relevant_tables,
            generated_sql=generated_sql,
            query_result=answer_table
        )
    except Exception as e:
        logger.error(f"Failed to store chat history: {e}")

    synthesized_results = synthesize_results(
        answer_text=answer_text,
        answer_table=answer_table,
        generated_sql=generated_sql,
        debug_info=debug_info
    )

    synthesized_results = process_synthesized_results(db_data, synthesized_results, sql_queries, [], logger)

    synthesized_results = add_incomplete_data_warning(synthesized_results, [])

    # Explain the Query and store result
    prompt_explainer_prompt = "prompts/prompt_explainer.txt"
    prompt_explainer_id = prompt_explainer_wrapper(sql_queries, db_data, prompt_explainer_prompt)

    synthesized_results['response_metadata']['explanation'] = prompt_explainer_id.id

    return synthesized_results


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Data Insight API",
        timestamp=datetime.now().isoformat()
    )


@app.post("/data-insight")
async def data_insight_endpoint(request: DataInsightRequest):
    """Main endpoint for processing data insight requests"""
    try:
        # Extract parameters with defaults
        session_id = request.session_id or str(uuid.uuid4())
        user_query = request.user_query
        chat_id = request.chat_id or str(int(datetime.now().timestamp() * 1000))
        # Pass shared clients explicitly
        result = process_data_insight(session_id, user_query, chat_id, llm_client, query_executor)
        return result
    except Exception as e:
        logger.error(f"Error processing data insight request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "CanIAnswerThePrompt": False,
                "agent_name": "data_insight",
                "error": "Internal server error occurred while processing your request",
                "response_metadata": {
                    "answer_img": [],
                    "answer_table": [],
                    "explanation": ""
                }
            }
        )


# Non-essential endpoints removed for production - keeping only core data-insight functionality


if __name__ == "__main__":
    # Configuration
    port = int(os.getenv('PORT', 8086))
    host = os.getenv('HOST', '0.0.0.0')
    reload = os.getenv('DEBUG', 'false').lower() == 'true'

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )