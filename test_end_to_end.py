"""
End-to-end test to debug the exact issue you're facing
"""

import pandas as pd

from core.utils.pandasai_helper import get_pandasai_helper
from core.utils.query_context_manager import get_query_context_manager


def test_exact_scenario():
    """Test the exact two queries you mentioned"""
    
    print("END-TO-END TEST: Your Exact Scenario")
    print("=" * 50)
    
    session_id = "test_session"
    context_manager = get_query_context_manager()
    pandasai_helper = get_pandasai_helper()
    
    # Clear any existing data for this session
    pandasai_helper.clear_session(session_id)
    
    print("\n" + "="*70)
    print("QUERY 1: 'Give me list of employees with salary above 60000 along with their department'")
    print("="*70)
    
    # Simulate first query execution and result storage
    query1 = "Give me list of employees with salary above 60000 along with their department"
    
    # Simulate database result (what would come from SQL execution)
    all_employees = pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Wilson', 'Charlie Brown', 'Diana Prince', 'Eve Adams', 'Frank Miller'],
        'salary': [85000, 92000, 78000, 105000, 55000, 67000, 45000, 98000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'HR', 'Finance', 'HR', 'Sales']
    })
    
    # Filter for salary > 60000 (what the SQL would return)
    query1_result = all_employees[all_employees['salary'] > 60000]
    
    print(f"Query 1 Result: {len(query1_result)} employees found")
    print(query1_result.to_string(index=False))
    
    # Create and store context for query 1
    context1 = context_manager.create_context(
        session_id=session_id,
        original_query=query1,
        processed_query=query1,
        tables_referenced=["employees"],
        generated_sql="SELECT employee_id, name, salary, department FROM employees WHERE salary > 60000",
        result_df=query1_result
    )
    
    # Store the dataframe with context
    success1 = pandasai_helper.create_smart_dataframe(query1_result, session_id, context1)
    print(f"\nStored Query 1 context and data: {success1}")
    print(f"Query type: {context1.query_type}")
    print(f"Entities: {context1.key_entities}")
    
    print("\n" + "="*70)
    print("QUERY 2: 'Give me list of employees with salary above 90000 along with their department'")
    print("="*70)
    
    query2 = "Give me list of employees with salary above 90000 along with their department"
    
    # Test follow-up detection
    print("Testing follow-up detection...")
    
    try:
        followup_result = pandasai_helper.can_answer_query(query2, session_id)
        print(f"Raw result from can_answer_query: {followup_result}")
        print(f"Type: {type(followup_result)}")
        
        if isinstance(followup_result, tuple) and len(followup_result) == 3:
            can_answer, relevant_context, confidence_score = followup_result
            print(f"✅ Properly unpacked: can_answer={can_answer}, confidence={confidence_score:.3f}")
            
            if relevant_context:
                print(f"✅ Relevant context found: {relevant_context.original_query}")
            else:
                print("❌ No relevant context")
                
            if can_answer and relevant_context and confidence_score >= 0.5:
                print(f"\n🎯 SHOULD BE DETECTED AS FOLLOW-UP (confidence {confidence_score:.3f} >= 0.5)")
                
                # Try to process it
                print("Attempting to process as follow-up...")
                result, error = pandasai_helper.query_dataframe(session_id, query2)
                
                if result is not None:
                    print("✅ SUCCESS: Follow-up query answered")
                    print(f"Result type: {type(result)}")
                    print(f"Result preview: {str(result)[:200]}...")
                    
                    # Check if it's the expected answer (employees with salary > 90000)
                    expected_result = query1_result[query1_result['salary'] > 90000]
                    print(f"\nExpected {len(expected_result)} employees with salary > 90000:")
                    print(expected_result.to_string(index=False))
                    
                else:
                    print(f"❌ FAILED: {error}")
            else:
                print("❌ NOT DETECTED AS FOLLOW-UP:")
                print(f"   - can_answer: {can_answer}")
                print(f"   - has_context: {relevant_context is not None}")
                print(f"   - confidence: {confidence_score:.3f} (needs >= 0.5)")
                
        else:
            print(f"❌ Unexpected result format: {followup_result}")
            
    except Exception as e:
        print(f"❌ ERROR in follow-up detection: {e}")
        import traceback
        traceback.print_exc()
    
    # Test what happens if we treat it as a fresh query
    print("\n" + "="*50)
    print("COMPARISON: If treated as fresh query...")
    print("="*50)
    
    fresh_result = all_employees[all_employees['salary'] > 90000]
    print(f"Fresh query would return {len(fresh_result)} employees:")
    print(fresh_result.to_string(index=False))

if __name__ == "__main__":
    test_exact_scenario()