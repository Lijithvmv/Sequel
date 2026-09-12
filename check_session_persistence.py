"""
Simple test to check if the session data persists between queries
"""

import pandas as pd

from core.utils.pandasai_helper import get_pandasai_helper
from core.utils.query_context_manager import get_query_context_manager


def check_session_data():
    print("TESTING SESSION DATA PERSISTENCE")
    print("=" * 40)
    
    session_id = "persistent_test"
    context_manager = get_query_context_manager()
    pandasai_helper = get_pandasai_helper()
    
    # Clear session
    pandasai_helper.clear_session(session_id)
    
    print(f"Session ID: {session_id}")
    print(f"Available sessions before: {pandasai_helper.get_available_sessions()}")
    
    # Simulate first query
    query1 = "Give me list of employees with salary above 60000 along with their department"
    
    data1 = pd.DataFrame({
        'employee_id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'salary': [70000, 80000, 90000],
        'department': ['Eng', 'Sales', 'HR']
    })
    
    context1 = context_manager.create_context(
        session_id=session_id,
        original_query=query1,
        processed_query=query1,
        tables_referenced=["employees"],
        generated_sql="SELECT * FROM employees WHERE salary > 60000",
        result_df=data1
    )
    
    # Store data
    success = pandasai_helper.create_smart_dataframe(data1, session_id, context1)
    print(f"Storage success: {success}")
    print(f"Available sessions after storage: {pandasai_helper.get_available_sessions()}")
    
    # Check if we can retrieve context
    recent_context = pandasai_helper.get_query_context(session_id)
    print(f"Can retrieve context: {recent_context is not None}")
    if recent_context:
        print(f"Retrieved context query: {recent_context.original_query}")
    
    # Test follow-up detection
    query2 = "Give me list of employees with salary above 90000 along with their department"
    result = pandasai_helper.can_answer_query(query2, session_id)
    
    print(f"Follow-up detection result: {result}")
    
    if isinstance(result, tuple) and len(result) == 3:
        can_answer, context, confidence = result
        print(f"  Can answer: {can_answer}")
        print(f"  Confidence: {confidence}")
        print(f"  Has context: {context is not None}")
    
    # Test actual query processing
    if isinstance(result, tuple) and result[0]:
        print("Testing actual query processing...")
        query_result, error = pandasai_helper.query_dataframe(session_id, query2)
        print(f"Query result: {query_result}")
        print(f"Error: {error}")

if __name__ == "__main__":
    check_session_data()