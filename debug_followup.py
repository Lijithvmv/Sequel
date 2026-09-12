"""
Simple debug script to test the exact scenario you mentioned
"""

import pandas as pd

from core.utils.pandasai_helper import get_pandasai_helper
from core.utils.query_context_manager import get_query_context_manager


def debug_your_scenario():
    """Test the exact scenario you mentioned"""
    
    context_manager = get_query_context_manager()
    pandasai_helper = get_pandasai_helper()
    
    session_id = "debug_session"
    
    print("DEBUGGING YOUR SPECIFIC SCENARIO")
    print("=" * 50)
    
    # Simulate the first query result
    print("\n1. FIRST QUERY: 'Give me list of employees with salary above 60000 along with their department'")
    
    first_data = {
        'employee_id': [1, 2, 3, 4, 5, 6],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana'],
        'salary': [65000, 72000, 85000, 95000, 58000, 105000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'HR', 'Sales']
    }
    first_df = pd.DataFrame(first_data)
    
    # Filter to simulate the actual query result (salary > 60000)
    filtered_df = first_df[first_df['salary'] > 60000]
    
    print(f"   Result: {len(filtered_df)} employees found")
    print(f"   Columns: {list(filtered_df.columns)}")
    
    # Create context for first query
    first_context = context_manager.create_context(
        session_id=session_id,
        original_query="Give me list of employees with salary above 60000 along with their department",
        processed_query="Give me list of employees with salary above 60000 along with their department",
        tables_referenced=["employees"],
        generated_sql="SELECT employee_id, name, salary, department FROM employees WHERE salary > 60000",
        result_df=filtered_df
    )
    
    # Store the dataframe and context
    success = pandasai_helper.create_smart_dataframe(filtered_df, session_id, first_context)
    print(f"   Storage success: {success}")
    print(f"   Query type detected: {first_context.query_type}")
    print(f"   Entities extracted: {first_context.key_entities}")
    
    # Now test the second query
    print("\n2. SECOND QUERY: 'Give me list of employees with salary above 90000 along with their department'")
    
    second_query = "Give me list of employees with salary above 90000 along with their department"
    
    # Test follow-up detection
    can_answer, relevant_context, confidence = pandasai_helper.can_answer_query(second_query, session_id)
    
    print(f"   Can answer as follow-up: {can_answer}")
    print(f"   Confidence score: {confidence:.3f}")
    
    if relevant_context:
        print(f"   Relevant context found: {relevant_context.original_query}")
        print(f"   Context query type: {relevant_context.query_type}")
        print(f"   Context entities: {relevant_context.key_entities}")
    else:
        print("   No relevant context found")
    
    # Analyze why it might not be detecting correctly
    print("\n3. DETAILED ANALYSIS:")
    
    # Check entity overlap
    current_entities = context_manager.extract_entities(second_query)
    print(f"   Current query entities: {current_entities}")
    
    if relevant_context:
        entity_overlap = set(current_entities) & set(relevant_context.key_entities)
        print(f"   Entity overlap: {entity_overlap}")
        
        overlap_ratio = len(entity_overlap) / len(relevant_context.key_entities) if relevant_context.key_entities else 0
        print(f"   Overlap ratio: {overlap_ratio:.3f}")
    
    # Check pattern matching
    followup_score = 0
    query_lower = second_query.lower()
    
    # Check for follow-up patterns
    patterns_found = []
    if 'above' in query_lower:
        patterns_found.append('above')
        followup_score += 0.3
    if 'employees' in query_lower:
        patterns_found.append('employees')
    if 'salary' in query_lower:
        patterns_found.append('salary')
    if 'department' in query_lower:
        patterns_found.append('department')
    
    print(f"   Patterns found: {patterns_found}")
    print(f"   Basic followup score: {followup_score}")
    
    # Test if the query can be processed
    if can_answer and confidence >= 0.5:
        print("\n4. ATTEMPTING TO PROCESS AS FOLLOW-UP:")
        try:
            result, error = pandasai_helper.query_dataframe(session_id, second_query)
            if result is not None:
                print(f"   SUCCESS: {str(result)[:100]}...")
            else:
                print(f"   FAILED: {error}")
        except Exception as e:
            print(f"   ERROR: {e}")
    else:
        print(f"\n4. NOT PROCESSING (confidence {confidence:.3f} < 0.5 or can't answer: {can_answer})")
        
        # Let's see what the threshold should be
        print("   Breakdown of confidence scoring:")
        print(f"   - Entity overlap would add: {0.4 * overlap_ratio if relevant_context else 0:.3f}")
        print(f"   - Pattern match adds: {0.3 if 'above' in query_lower else 0:.3f}")
        print(f"   - Column mentions would add: {0.2 * min(1.0, sum(1 for col in first_context.column_names if col.lower() in query_lower) / len(first_context.column_names)):.3f}")

if __name__ == "__main__":
    try:
        debug_your_scenario()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()