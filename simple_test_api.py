#!/usr/bin/env python3
"""
Simplified API test to isolate the 500 error
"""
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Simple Test API")

class DataInsightRequest(BaseModel):
    session_id: Optional[str] = None
    user_query: str
    chat_id: Optional[str] = None

@app.post("/data-insight")
async def simple_data_insight(request: DataInsightRequest):
    """Simplified data insight endpoint for testing"""
    try:
        # Import and test components one by one
        from core.actions.queryexecutor import QueryExecutor
        from core.connectors.LLMClient import LLMClient
        from core.logger import get_logger, set_session_id
        
        # Initialize
        session_id = request.session_id or str(uuid.uuid4())
        user_query = request.user_query
        request.chat_id or str(int(datetime.now().timestamp() * 1000))
        
        set_session_id(session_id)
        get_logger(__name__)
        
        # Test LLM
        llm_client = LLMClient()
        query_executor = QueryExecutor()
        
        # Test table selector
        from core.actions.tableselector import TableSelector
        table_selector = TableSelector(
            llm_client=llm_client,
            table_descriptions_file="table_descriptions.csv",
            instruction_file="table_selector_instruction.txt"
        )
        
        selected_tables = table_selector.run(user_query)
        
        if not selected_tables.get('relevant_tables_found', False):
            return {
                "CanIAnswerThePrompt": False,
                "agent_name": "data_insight",
                "answer_text": "No relevant tables found for your query.",
                "response_metadata": {
                    "answer_img": [],
                    "answer_table": [],
                    "explanation": ""
                }
            }
        
        # Test query generator
        from core.actions.querygenerator import QueryGenerator
        query_generator = QueryGenerator(
            llm_client=llm_client,
            single_table_descriptions_file="dummy.json",
            query_generator_prompt="prompts/query_generator.txt"
        )
        
        sql_queries, flag, debug_info = query_generator.generate_sql_queries(
            user_query,
            selected_tables['relevant_tables'],
            error=""
        )
        
        if flag == 0:
            return {
                "CanIAnswerThePrompt": False,
                "agent_name": "data_insight",
                "answer_text": "Could not generate SQL queries.",
                "response_metadata": {
                    "answer_img": [],
                    "answer_table": [],
                    "explanation": ""
                }
            }
        
        # Execute queries
        query_executor.connect()
        answer_tables = []
        generated_sqls = []
        
        try:
            for table, queries in sql_queries.items():
                for title, sql in queries.items():
                    df, error = query_executor.execute_single_query(sql)
                    generated_sqls.append(sql)
                    if not df.empty:
                        answer_tables.append([df.columns.tolist()] + df.values.tolist())
        finally:
            query_executor.close_connection()
        
        answer_table = next((tbl for tbl in answer_tables if len(tbl) > 1), [])
        
        return {
            "CanIAnswerThePrompt": True,
            "agent_name": "data_insight",
            "answer_text": f"Found {len(answer_table)-1 if answer_table else 0} results",
            "response_metadata": {
                "answer_img": [],
                "answer_table": answer_table,
                "explanation": "Simplified test response",
                "generated_sql": "\n".join(generated_sqls)
            }
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in simple API: {e}")
        print(f"Full traceback:\n{error_details}")
        
        return {
            "CanIAnswerThePrompt": False,
            "agent_name": "data_insight",
            "error": f"Error: {str(e)}",
            "error_details": error_details,
            "response_metadata": {
                "answer_img": [],
                "answer_table": [],
                "explanation": ""
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_test_api:app", host="0.0.0.0", port=8084, reload=True)