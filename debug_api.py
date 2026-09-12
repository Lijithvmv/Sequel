#!/usr/bin/env python3
"""
Debug script to isolate the 500 error in the API
"""
import os
import sys
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

def debug_process_data_insight():
    try:
        print("=== Starting Debug Process ===")
        
        # Import modules one by one
        print("1. Importing basic modules...")
        from core.logger import get_logger, set_session_id
        print("   + Logger imported")
        
        from core.connectors.LLMClient import LLMClient
        print("   + LLMClient imported")
        
        from core.actions.queryexecutor import QueryExecutor
        print("   + QueryExecutor imported")
        
        # Initialize components
        print("2. Initializing components...")
        session_id = "debug-session"
        user_query = "Show me all employees"
        
        set_session_id(session_id)
        logger = get_logger(__name__)
        print("   + Logger initialized")
        
        llm_client = LLMClient()
        QueryExecutor()
        print("   + Clients initialized")
        
        # Test database operations
        print("3. Testing database operations...")
        from core.utils.database import read_session_chat_data
        
        session_df = read_session_chat_data({'session_id': session_id}, logger)
        print(f"   + Session data read: {session_df.shape}")
        
        # Test table selector
        print("4. Testing table selector...")
        from core.actions.tableselector import TableSelector
        
        table_selector = TableSelector(
            llm_client=llm_client,
            table_descriptions_file="table_descriptions.csv",
            instruction_file="table_selector_instruction.txt",
            use_ai=True,
            hybrid_mode="ai_first",
            ai_timeout=5.0,
            fallback_threshold=0.3
        )
        print("   + TableSelector created")
        
        selected_tables = table_selector.run(user_query)
        print(f"   + Tables selected: {selected_tables.get('relevant_tables', [])}")
        
        # Test query generator
        print("5. Testing query generator...")
        from core.actions.querygenerator import QueryGenerator
        
        query_generator = QueryGenerator(
            llm_client=llm_client,
            single_table_descriptions_file="dummy.json",
            query_generator_prompt="prompts/query_generator.txt"
        )
        print("   + QueryGenerator created")
        
        if selected_tables.get('relevant_tables_found', False):
            sql_queries, flag, debug_info = query_generator.generate_sql_queries(
                user_query,
                selected_tables['relevant_tables'],
                error=""
            )
            print(f"   + SQL queries generated: {flag}")
            print(f"   + Debug info: {debug_info}")
        else:
            print("   ! No relevant tables found")
        
        print("=== Debug Process Completed Successfully ===")
        return True
        
    except Exception as e:
        print("\n=== ERROR DETECTED ===")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_process_data_insight()
    if success:
        print("\n[SUCCESS] All components working correctly!")
    else:
        print("\n[ERROR] Found the error causing 500 status!")