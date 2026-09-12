"""
Chat History Based Follow-up Detection and PandasAI Integration

This module provides efficient follow-up detection using stored chat history
and converts answer_table data to DataFrames for PandasAI querying.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

class ChatHistoryFollowUpDetector:
    """Detects follow-up queries using chat history and provides DataFrame querying"""
    
    def __init__(self):
        self.followup_patterns = [
            # Direct follow-up indicators
            r'\b(above|below|greater|less|more|fewer)\s+than\s+\d+',
            r'\b(top|bottom|first|last)\s+\d+',
            r'\b(those|these|them|that)\s+(with|above|below)',
            r'\b(now|then|also|additionally)\s+(show|give|filter)',
            r'\bfrom\s+(the|that|this)\s+(result|data|list)',
            r'\bshow\s+(me\s+)?(those|these|them)',
            
            # Entity continuity patterns
            r'\b(employees?|workers?|staff)\s+(with|above|below)',
            r'\b(salary|income|wage|pay)\s+(above|below|greater|less)',
            r'\b(department|team|division)',
            r'salary\s+above\s+\d+',
            r'salary\s+below\s+\d+',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.followup_patterns]
    
    def detect_followup(self, current_query: str, chat_history: List[Dict]) -> Tuple[bool, Optional[Dict], float]:
        """
        Detect if current query is a follow-up based on chat history
        
        Returns:
            (is_followup, previous_context, confidence_score)
        """
        if not chat_history:
            return False, None, 0.0
        
        # Get the most recent chat entry
        recent_chat = chat_history[0]  # Most recent first from chat manager
        
        # Check if recent chat has the required data
        if not self._has_valid_context(recent_chat):
            return False, None, 0.0
        
        # Calculate follow-up confidence
        confidence = self._calculate_followup_confidence(current_query, recent_chat)
        
        is_followup = confidence >= 0.5  # Adjusted threshold for better detection
        
        logger.info(f"Follow-up detection: '{current_query[:50]}...' -> Confidence: {confidence:.2f}, Is follow-up: {is_followup}")
        
        return is_followup, recent_chat if is_followup else None, confidence
    
    def _has_valid_context(self, chat_entry: Dict) -> bool:
        """Check if chat entry has valid context for follow-up processing"""
        required_fields = ['user_query', 'generated_sql', 'tables_referenced']
        return all(chat_entry.get(field) for field in required_fields)
    
    def _calculate_followup_confidence(self, current_query: str, previous_chat: Dict) -> float:
        """Calculate confidence score for follow-up detection"""
        confidence = 0.0
        current_lower = current_query.lower()
        previous_query = previous_chat.get('user_query', '').lower()
        
        # 1. Pattern matching (40% weight)
        pattern_matches = sum(1 for pattern in self.compiled_patterns if pattern.search(current_lower))
        if pattern_matches > 0:
            confidence += 0.4 * min(pattern_matches / 3, 1.0)
        
        # 2. Entity overlap (30% weight)
        current_entities = self._extract_entities(current_lower)
        previous_entities = self._extract_entities(previous_query)
        
        if current_entities and previous_entities:
            overlap = len(set(current_entities) & set(previous_entities))
            overlap_ratio = overlap / max(len(previous_entities), 1)
            confidence += 0.3 * overlap_ratio
        
        # 3. Table/column references (20% weight)
        tables_referenced = previous_chat.get('tables_referenced', [])
        table_mentions = sum(1 for table in tables_referenced if table.lower() in current_lower)
        if table_mentions > 0:
            confidence += 0.2 * min(table_mentions / len(tables_referenced), 1.0)
        
        # 4. Time proximity (10% weight) - recent queries more likely to be follow-ups
        try:
            timestamp = previous_chat.get('timestamp', '')
            if timestamp:
                chat_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00').replace('+00:00', ''))
                time_diff = datetime.now() - chat_time
                if time_diff < timedelta(minutes=5):
                    confidence += 0.1
                elif time_diff < timedelta(minutes=30):
                    confidence += 0.05
        except:
            pass
        
        return min(confidence, 1.0)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract business entities from query"""
        entities = []
        entity_patterns = {
            'employees': r'\b(employee|employees|worker|workers|staff|person|people)\b',
            'salary': r'\b(salary|salaries|income|wage|wages|pay|compensation)\b',
            'department': r'\b(department|dept|departments|division|team|unit)\b',
            'amount': r'\b(amount|value|total|sum|count)\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                entities.append(entity_type)
        
        return entities
    
    def convert_answer_table_to_dataframe(self, answer_table: List[List]) -> Optional[pd.DataFrame]:
        """
        Convert answer_table format to pandas DataFrame
        
        Args:
            answer_table: List of lists where first list is headers, rest are rows
        
        Returns:
            pandas DataFrame or None if conversion fails
        """
        if not answer_table or len(answer_table) < 2:
            logger.warning("Invalid answer_table format for DataFrame conversion")
            return None
        
        try:
            headers = answer_table[0]
            rows = answer_table[1:]
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Try to infer and convert data types
            df = self._optimize_dataframe_types(df)
            
            logger.info(f"Converted answer_table to DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting answer_table to DataFrame: {e}")
            return None
    
    def _optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame column types for better PandasAI performance"""
        try:
            for col in df.columns:
                # Try to convert numeric columns
                if df[col].dtype == 'object':
                    # Check if all values can be converted to numeric
                    try:
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_series.isna().all():
                            # If most values are numeric, convert the column
                            non_null_ratio = numeric_series.notna().sum() / len(df)
                            if non_null_ratio > 0.8:  # 80% of values are numeric
                                df[col] = numeric_series
                    except:
                        pass
            
            return df
            
        except Exception as e:
            logger.warning(f"Error optimizing DataFrame types: {e}")
            return df
    
    def create_pandasai_agent(self, df: pd.DataFrame, context: str = "", save_path: str = "./sessions") -> Any:
        """
        Create a simple wrapper that holds the DataFrame and can create SmartDataframe on demand
        
        Args:
            df: pandas DataFrame
            context: Context description for the DataFrame
            save_path: Path to save charts/logs
            
        Returns:
            Simple wrapper object with DataFrame access
        """
        try:
            # Create a simple wrapper that holds the DataFrame and lazy-loads SmartDataframe
            class DataframeWrapper:
                def __init__(self, dataframe, save_path):
                    self.dataframe = dataframe
                    self.save_path = save_path
                    self._smart_df = None
                    
                def get_smartdataframe(self):
                    if self._smart_df is None:
                        from pandasai import SmartDataframe

                        from core.config import set_default_config
                        from core.connectors.LLMClient import LLMClient
                        from core.utils.pandasAIWrapper import CustomLLMWrapper
                        
                        config = set_default_config()
                        llm_client = LLMClient(config)
                        llm_wrapper = CustomLLMWrapper(client=llm_client, config=config)
                        
                        self._smart_df = SmartDataframe(
                            self.dataframe,
                            config={
                                "llm": llm_wrapper,
                                "save_charts": True,
                                "save_charts_path": self.save_path,
                                "enable_cache": False,
                                "save_logs": False
                            }
                        )
                    return self._smart_df
            
            wrapper = DataframeWrapper(df, save_path)
            logger.info(f"Created DataFrame wrapper with {len(df)} rows for querying")
            return wrapper
            
        except ImportError as e:
            logger.warning(f"PandasAI components not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating PandasAI agent: {e}")
            return None
    
    def query_pandasai_agent(self, agent: Any, query: str) -> Tuple[Optional[Any], Optional[str]]:
        """
        Query the DataframeWrapper with SmartDataframe fallback
        
        Args:
            agent: DataframeWrapper instance
            query: Natural language query
            
        Returns:
            (result, error) - result or None if error
        """
        try:
            if agent is None:
                return None, "DataframeWrapper not available"
            
            logger.info(f"Querying DataframeWrapper with: {query}")
            
            # First try direct pandas operations for common patterns (faster)
            try:
                result = self._query_with_direct_pandas(query, agent.dataframe)
                if result is not None:
                    logger.info("Used direct pandas fallback successfully")
                    return result, None
            except Exception as direct_error:
                logger.warning(f"Direct pandas attempt failed: {direct_error}")
            
            # If direct pandas couldn't handle it, try SmartDataframe
            try:
                smart_df = agent.get_smartdataframe()
                if smart_df is not None:
                    logger.info("Using SmartDataframe for complex query")
                    result = smart_df.chat(query)
                    
                    logger.info(f"SmartDataframe query successful: {query[:50]}...")
                    
                    # Convert result to proper format for response
                    if isinstance(result, pd.DataFrame):
                        # If result is a DataFrame, convert to answer_table format
                        if len(result) > 0:
                            answer_table = [result.columns.tolist()] + result.values.tolist()
                            return answer_table, None
                        else:
                            return "No matching records found", None
                    else:
                        # String or other response
                        return str(result), None
                else:
                    return None, "SmartDataframe creation failed"
                    
            except Exception as smart_df_error:
                logger.error(f"SmartDataframe query failed: {smart_df_error}")
                
                # Final fallback: basic DataFrame info with helpful suggestions
                try:
                    df = agent.dataframe
                    
                    # Provide contextual help based on available columns
                    available_cols = df.columns.tolist()
                    suggestions = []
                    
                    if any('salary' in col.lower() for col in available_cols):
                        suggestions.append("salary above/below [amount]")
                    if any('department' in col.lower() for col in available_cols):
                        suggestions.append("department filtering")
                    if any('name' in col.lower() for col in available_cols):
                        suggestions.append("name search")
                    
                    suggestion_text = f"Try: {', '.join(suggestions)}" if suggestions else "Try simpler queries like 'count', 'top 5', etc."
                    
                    fallback_result = (f"Unable to process complex query. Dataset contains {len(df)} records "
                                     f"with columns: {', '.join(available_cols)}. {suggestion_text}")
                    return fallback_result, None
                except:
                    return None, f"All query methods failed. Last error: {smart_df_error}"
            
        except Exception as e:
            error_msg = f"DataframeWrapper query failed: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _query_with_direct_pandas(self, query: str, df: pd.DataFrame) -> Optional[Any]:
        """
        Handle common queries with direct pandas operations for better performance and reliability
        """
        query_lower = query.lower()
        
        try:
            # Pattern 1: Salary filtering with "above" or "greater than"
            if match := re.search(r'salary\s+(above|greater\s+than|>)\s+(\d+)', query_lower):
                value = float(match.group(2))
                result_df = df[df['salary'] > value]
                if not result_df.empty:
                    return [result_df.columns.tolist()] + result_df.values.tolist()
                else:
                    return "No employees found with salary above " + str(value)
            
            # Pattern 2: Salary filtering with "below" or "less than"
            if match := re.search(r'salary\s+(below|less\s+than|<)\s+(\d+)', query_lower):
                value = float(match.group(2))
                result_df = df[df['salary'] < value]
                if not result_df.empty:
                    return [result_df.columns.tolist()] + result_df.values.tolist()
                else:
                    return "No employees found with salary below " + str(value)
            
            # Pattern 3: Count queries
            if any(word in query_lower for word in ['how many', 'count', 'total']):
                return f"Total count: {len(df)} employees"
            
            # Pattern 4: Top N records
            if match := re.search(r'top\s+(\d+)', query_lower):
                n = int(match.group(1))
                result_df = df.head(n)
                return [result_df.columns.tolist()] + result_df.values.tolist()
            
            # Pattern 5: Department filtering (flexible patterns)
            dept_patterns = [
                r'department\s*[=:]?\s*["\']?(\w+)["\']?',
                r'in\s+(\w+)\s+department',
                r'from\s+(\w+)\s+department',
                r'(\w+)\s+department'
            ]
            
            for pattern in dept_patterns:
                if match := re.search(pattern, query_lower):
                    dept = match.group(1)
                    # Check for department_name or department column
                    dept_col = None
                    for col in df.columns:
                        if 'department' in col.lower():
                            dept_col = col
                            break
                    
                    if dept_col:
                        result_df = df[df[dept_col].str.contains(dept, case=False, na=False)]
                        if not result_df.empty:
                            return [result_df.columns.tolist()] + result_df.values.tolist()
                        else:
                            return f"No employees found in {dept} department"
            
            # Pattern 6: Name/person filtering
            if match := re.search(r'(show|find|get)\s+(\w+)(?:\s+salary|\s+department|\s+info)?', query_lower):
                name = match.group(2)
                # Look for name column
                name_col = None
                for col in df.columns:
                    if 'name' in col.lower():
                        name_col = col
                        break
                
                if name_col:
                    result_df = df[df[name_col].str.contains(name, case=False, na=False)]
                    if not result_df.empty:
                        return [result_df.columns.tolist()] + result_df.values.tolist()
                    else:
                        return f"No employee found with name containing '{name}'"
            
            # Pattern 7: Sorting queries
            if 'highest' in query_lower and 'salary' in query_lower:
                if 'salary' in df.columns:
                    result_df = df.nlargest(1, 'salary')
                    return [result_df.columns.tolist()] + result_df.values.tolist()
            
            if 'lowest' in query_lower and 'salary' in query_lower:
                if 'salary' in df.columns:
                    result_df = df.nsmallest(1, 'salary')
                    return [result_df.columns.tolist()] + result_df.values.tolist()
            
            # Pattern 8: Average/mean queries
            if any(word in query_lower for word in ['average', 'mean']) and 'salary' in query_lower:
                if 'salary' in df.columns:
                    avg_salary = df['salary'].mean()
                    return f"Average salary: ${avg_salary:,.2f}"
            
        except Exception as e:
            logger.warning(f"Direct pandas query failed: {e}")
            return None
        
        # Return None if no pattern matched - will fallback to SmartDataframe
        return None

# Global instance
_followup_detector = None

def get_followup_detector() -> ChatHistoryFollowUpDetector:
    """Get the global follow-up detector instance"""
    global _followup_detector
    if _followup_detector is None:
        _followup_detector = ChatHistoryFollowUpDetector()
    return _followup_detector