"""
Test script to demonstrate the enhanced follow-up detection logic.
This script simulates real scenarios to show how the new system works.
"""

import pandas as pd

from core.utils.pandasai_helper import get_pandasai_helper
from core.utils.query_context_manager import get_query_context_manager


def test_followup_scenarios():
    """Test various follow-up scenarios"""
    
    context_manager = get_query_context_manager()
    pandasai_helper = get_pandasai_helper()
    
    # Test session
    session_id = "test_session_123"
    
    print("="*80)
    print("TESTING ENHANCED FOLLOW-UP DETECTION LOGIC")
    print("="*80)
    
    # Scenario 1: Initial query about employees
    print("\n1. INITIAL QUERY: 'Give me list of employees with their salary and department'")
    
    # Simulate initial query result
    initial_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Wilson', 'Charlie Brown'],
        'salary': [85000, 92000, 78000, 105000, 67000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'HR']
    }
    initial_df = pd.DataFrame(initial_data)
    
    # Create initial context
    initial_context = context_manager.create_context(
        session_id=session_id,
        original_query="Give me list of employees with their salary and department",
        processed_query="Give me list of employees with their salary and department",
        tables_referenced=["employees"],
        generated_sql="SELECT employee_id, name, salary, department FROM employees",
        result_df=initial_df
    )
    
    # Store the context and dataframe
    pandasai_helper.create_smart_dataframe(initial_df, session_id, initial_context)
    
    print(f"   - Stored {len(initial_df)} employees with context")
    print(f"   - Query type: {initial_context.query_type}")
    print(f"   - Key entities: {initial_context.key_entities}")
    
    # Test follow-up scenarios
    followup_queries = [
        # Strong follow-ups (should be detected)
        "Show me employees with salary above 80000",
        "Give me those with salary above 90000 along with their department",
        "Show employees with salary above 70000",
        "Who are the top 3 highest paid employees?",
        "Show me the average salary by department",
        "Filter those employees earning more than 85000",
        
        # Weak follow-ups (might be detected based on entities)
        "Count the total number of employees",
        "What departments do we have?",
        
        # Non-follow-ups (should NOT be detected)
        "Show me all products in inventory",
        "What are the sales figures for last month?",
        "Give me customer satisfaction scores",
    ]
    
    print(f"\n2. TESTING {len(followup_queries)} FOLLOW-UP QUERIES:")
    print("-" * 60)
    
    for i, query in enumerate(followup_queries, 1):
        print(f"\n{i:2d}. Query: '{query}'")
        
        # Test follow-up detection
        can_answer, relevant_context, confidence = pandasai_helper.can_answer_query(query, session_id)
        
        print(f"    -> Can answer: {can_answer}")
        print(f"    -> Confidence: {confidence:.2f}")
        print(f"    -> Relevant context: {relevant_context.original_query[:40] if relevant_context else 'None'}...")
        
        # Categorize result
        if confidence >= 0.7:
            category = "🟢 STRONG FOLLOW-UP"
        elif confidence >= 0.5:
            category = "🟡 WEAK FOLLOW-UP"
        elif confidence >= 0.3:
            category = "🟠 POSSIBLE FOLLOW-UP"
        else:
            category = "🔴 NOT A FOLLOW-UP"
        
        print(f"    -> Category: {category}")
        
        # If it's a follow-up, try to process it
        if can_answer and confidence >= 0.5:
            try:
                result, error = pandasai_helper.query_dataframe(session_id, query)
                if result is not None:
                    print(f"    -> ✅ Successfully answered: {str(result)[:100]}...")
                else:
                    print(f"    -> ❌ Failed to answer: {error}")
            except Exception as e:
                print(f"    -> ❌ Error processing: {e}")

def test_context_analysis():
    """Test the context analysis features"""
    
    print("\n" + "="*80)
    print("TESTING CONTEXT ANALYSIS FEATURES")
    print("="*80)
    
    context_manager = get_query_context_manager()
    
    # Test entity extraction
    test_queries = [
        "Show me employees with salary above 50000",
        "Give me product sales by department",
        "What customers ordered more than 100 items?",
        "Find workers with high income in engineering team"
    ]
    
    print("\n1. ENTITY EXTRACTION TEST:")
    print("-" * 40)
    
    for query in test_queries:
        entities = context_manager.extract_entities(query)
        query_type = context_manager.determine_query_type(query)
        print(f"Query: '{query}'")
        print(f"  -> Entities: {entities}")
        print(f"  -> Type: {query_type}")
        print()

def test_followup_patterns():
    """Test follow-up pattern recognition"""
    
    print("\n2. FOLLOW-UP PATTERN RECOGNITION TEST:")
    print("-" * 45)
    
    context_manager = get_query_context_manager()
    
    # Create a dummy context
    session_id = "pattern_test"
    dummy_data = pd.DataFrame({'id': [1, 2], 'value': [100, 200]})
    
    dummy_context = context_manager.create_context(
        session_id=session_id,
        original_query="Show me all records",
        processed_query="Show me all records", 
        tables_referenced=["test_table"],
        generated_sql="SELECT * FROM test_table",
        result_df=dummy_data
    )
    
    context_manager.store_context(dummy_context)
    
    # Test various follow-up patterns
    pattern_tests = [
        # Direct references
        "Show me those with value above 150",
        "Filter these records by value",
        "Give me the top 5 from that result",
        
        # Indirect references  
        "Now show me the average",
        "Also include the count",
        "Additionally, sort by value",
        
        # Non-patterns
        "Show me different data",
        "What about other tables?",
    ]
    
    for pattern in pattern_tests:
        is_followup, context, confidence = context_manager.is_followup_query(pattern, session_id)
        print(f"'{pattern}' -> Follow-up: {is_followup}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    try:
        test_followup_scenarios()
        test_context_analysis() 
        test_followup_patterns()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()