"""
Simple data analyzer for follow-up questions when PandasAI is not available.
Provides basic data analysis capabilities without requiring external LLMs.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SimpleDataAnalyzer:
    """Simple data analyzer for basic follow-up questions"""
    
    def __init__(self):
        self.session_dataframes = {}
    
    def store_dataframe(self, df: pd.DataFrame, session_id: str, context: str = "") -> bool:
        """Store a dataframe for analysis"""
        try:
            self.session_dataframes[session_id] = {
                "dataframe": df.copy(),
                "created_at": datetime.now(),
                "context": context,
                "original_shape": df.shape
            }
            logger.info(f"Stored dataframe for session {session_id} with shape {df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error storing dataframe: {e}")
            return False
    
    def can_answer_query(self, query: str, session_id: str) -> bool:
        """Check if we can answer the query with simple operations"""
        if session_id not in self.session_dataframes:
            return False
        
        query_lower = query.lower()
        df = self.session_dataframes[session_id]["dataframe"]
        
        # Keywords that indicate simple data operations
        simple_keywords = [
            # Basic display
            "show", "display", "print", "view", "see", "list", "give me",
            # Filtering and sorting
            "top", "bottom", "first", "last", "head", "tail", "sort", "order",
            # Aggregations
            "count", "sum", "average", "mean", "max", "maximum", "min", "minimum",
            "total", "std", "median", "unique", "distinct",
            # Basic filtering
            "where", "filter", "greater", "less", "equal", "above", "below",
            # Basic info
            "shape", "size", "columns", "info", "describe", "summary",
            # Domain-specific
            "employees", "salary", "department", "income", "wage"
        ]
        
        # Check for basic keywords
        has_keywords = any(keyword in query_lower for keyword in simple_keywords)
        
        # Enhanced check: Can we identify column names and operations?
        if has_keywords:
            # Check if query mentions columns that exist in our dataframe
            column_matches = [col for col in df.columns if col.lower() in query_lower]
            
            # Check for numeric filtering operations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            salary_like_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['salary', 'income', 'wage', 'pay'])]
            
            # If it's a filtering query with numeric columns, we can probably handle it
            if any(op in query_lower for op in ['above', 'below', 'greater', 'less', '>', '<']):
                if column_matches or salary_like_cols:
                    return True
            
            return True
        
        return False
    
    def analyze_query(self, query: str, session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Analyze a query and return results"""
        if session_id not in self.session_dataframes:
            return None, "No data available for this session"
        
        df = self.session_dataframes[session_id]["dataframe"]
        query_lower = query.lower()
        
        try:
            # Enhanced filtering operations for salary/numeric columns
            if any(op in query_lower for op in ['above', 'below', 'greater', 'less', '>', '<']) and any(word in query_lower for word in ['salary', 'income', 'wage', 'pay', 'employees']):
                return self._handle_numeric_filtering(df, query), None
            
            # Basic display operations
            if any(word in query_lower for word in ["show", "display", "view", "print", "list", "give me"]):
                # Check if it's a filtering request first
                if any(op in query_lower for op in ['above', 'below', 'greater', 'less', '>', '<']):
                    return self._handle_numeric_filtering(df, query), None
                elif "top" in query_lower or "first" in query_lower:
                    n = self._extract_number(query, default=5)
                    result = df.head(n)
                    return self._format_dataframe_result(result, f"Top {n} rows"), None
                elif "bottom" in query_lower or "last" in query_lower:
                    n = self._extract_number(query, default=5)
                    result = df.tail(n)
                    return self._format_dataframe_result(result, f"Bottom {n} rows"), None
                else:
                    return self._format_dataframe_result(df.head(10), "First 10 rows"), None
            
            # Aggregation operations
            if "count" in query_lower:
                if "unique" in query_lower or "distinct" in query_lower:
                    result = df.nunique()
                    return f"Unique values per column:\n{result.to_string()}", None
                else:
                    return f"Total rows: {len(df)}", None
            
            if any(word in query_lower for word in ["sum", "total"]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols].sum()
                    return f"Sum of numeric columns:\n{result.to_string()}", None
                else:
                    return "No numeric columns found for sum operation", None
            
            if any(word in query_lower for word in ["average", "mean"]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols].mean()
                    return f"Average of numeric columns:\n{result.to_string()}", None
                else:
                    return "No numeric columns found for average operation", None
            
            if any(word in query_lower for word in ["max", "maximum"]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols].max()
                    return f"Maximum values:\n{result.to_string()}", None
                else:
                    return "No numeric columns found for max operation", None
            
            if any(word in query_lower for word in ["min", "minimum"]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols].min()
                    return f"Minimum values:\n{result.to_string()}", None
                else:
                    return "No numeric columns found for min operation", None
            
            # Info operations
            if any(word in query_lower for word in ["shape", "size"]):
                return f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns", None
            
            if "columns" in query_lower:
                return f"Columns: {', '.join(df.columns.tolist())}", None
            
            if "describe" in query_lower or "summary" in query_lower:
                result = df.describe()
                return self._format_dataframe_result(result, "Statistical Summary"), None
            
            if "info" in query_lower:
                info_str = f"""Dataset Information:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data types: {df.dtypes.value_counts().to_dict()}
- Missing values: {df.isnull().sum().sum()}"""
                return info_str, None
            
            # Sorting operations
            if "sort" in query_lower or "order" in query_lower:
                # Try to identify column name
                col_name = self._extract_column_name(query, df.columns)
                if col_name:
                    ascending = "desc" not in query_lower and "descending" not in query_lower
                    result = df.sort_values(by=col_name, ascending=ascending).head(10)
                    direction = "ascending" if ascending else "descending"
                    return self._format_dataframe_result(result, f"Sorted by {col_name} ({direction})"), None
                else:
                    return "Please specify a column name to sort by", None
            
            # If no specific operation found, return basic info
            return f"I can help with basic operations on your data ({df.shape[0]} rows, {df.shape[1]} columns). Try asking about: show top 10, count, sum, average, sort by column, etc.", None
            
        except Exception as e:
            error_msg = f"Error analyzing query: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _extract_number(self, text: str, default: int = None) -> Optional[int]:
        """Extract a number from text, return default if not found"""
        numbers = re.findall(r'\d+', text)
        if numbers:
            # For filtering queries, return the largest number (likely the threshold)
            # For display queries, return the first number
            if any(op in text.lower() for op in ['above', 'below', 'greater', 'less', '>', '<']):
                return int(max(numbers, key=int))
            else:
                return int(numbers[0])
        return default if default is not None else 5
    
    def _extract_column_name(self, text: str, columns: List[str]) -> Optional[str]:
        """Try to extract a column name from the query text"""
        text_lower = text.lower()
        for col in columns:
            if col.lower() in text_lower:
                return col
        return None
    
    def _format_dataframe_result(self, df: pd.DataFrame, title: str) -> str:
        """Format a dataframe result as a string"""
        if df.empty:
            return f"{title}: No data to display"
        
        # Limit display size for readability
        max_rows = 20
        max_cols = 8
        
        display_df = df
        truncated_info = ""
        
        if len(df) > max_rows:
            display_df = df.head(max_rows)
            truncated_info += f" (showing first {max_rows} of {len(df)} rows)"
        
        if len(df.columns) > max_cols:
            display_df = display_df.iloc[:, :max_cols]
            truncated_info += f" (showing first {max_cols} of {len(df.columns)} columns)"
        
        result = f"{title}{truncated_info}:\n\n{display_df.to_string(index=True, max_rows=max_rows)}"
        return result
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about stored data for a session"""
        if session_id not in self.session_dataframes:
            return None
        
        data = self.session_dataframes[session_id]
        df = data["dataframe"]
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "created_at": data["created_at"].isoformat(),
            "context": data["context"],
            "original_shape": data["original_shape"]
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear stored data for a session"""
        if session_id in self.session_dataframes:
            del self.session_dataframes[session_id]
            logger.info(f"Cleared data for session {session_id}")
            return True
        return False
    
    def get_available_sessions(self) -> List[str]:
        """Get list of sessions with stored data"""
        return list(self.session_dataframes.keys())
    
    def _handle_numeric_filtering(self, df: pd.DataFrame, query: str) -> str:
        """Handle numeric filtering queries like 'employees with salary above X'"""
        try:
            query_lower = query.lower()
            
            # Extract numeric threshold
            threshold = self._extract_number(query)
            if threshold is None:
                return "Could not identify numeric threshold in query"
            
            # Identify the column to filter on
            target_column = None
            
            # PRIORITY 1: If query mentions 'salary', always use salary column first
            if 'salary' in query_lower:
                for col in df.columns:
                    if 'salary' in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                        target_column = col
                        break
            
            # PRIORITY 2: Look for explicitly mentioned columns
            if target_column is None:
                for col in df.columns:
                    if col.lower() in query_lower:
                        # Check if it's a numeric column
                        if pd.api.types.is_numeric_dtype(df[col]):
                            target_column = col
                            break
            
            # PRIORITY 3: Look for salary-like columns
            if target_column is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                salary_like_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['salary', 'income', 'wage', 'pay', 'amount'])]
                
                if salary_like_cols:
                    target_column = salary_like_cols[0]  # Use first salary-like column
            
            # PRIORITY 4: Use any numeric column as last resort
            if target_column is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_column = numeric_cols[0]
                else:
                    return "No numeric columns found for filtering"
            
            # Debug: Log which column is being used
            print(f"DEBUG: Available columns in dataframe: {list(df.columns)}")
            print(f"DEBUG: Dataframe shape: {df.shape}")
            print(f"DEBUG: Query mentions 'salary': {'salary' in query_lower}")
            print(f"DEBUG: Filtering by column '{target_column}' with threshold {threshold}")
            print("DEBUG: First few rows of dataframe:")
            print(df.head(2).to_string())
            
            # Apply the filter based on operation
            if any(op in query_lower for op in ['above', 'greater', '>']):
                filtered_df = df[df[target_column] > threshold]
                operation = f"above {threshold}"
            elif any(op in query_lower for op in ['below', 'less', '<']):
                filtered_df = df[df[target_column] < threshold]
                operation = f"below {threshold}"
            elif any(op in query_lower for op in ['equal', '=']):
                filtered_df = df[df[target_column] == threshold]
                operation = f"equal to {threshold}"
            else:
                # Default to above
                filtered_df = df[df[target_column] > threshold]
                operation = f"above {threshold}"
            
            if filtered_df.empty:
                return f"No records found with {target_column} {operation}"
            
            title = f"Records with {target_column} {operation} ({len(filtered_df)} rows)"
            return self._format_dataframe_result(filtered_df, title)
            
        except Exception as e:
            return f"Error processing filtering query: {str(e)}"
    

# Global instance
simple_analyzer = SimpleDataAnalyzer()

def get_simple_analyzer() -> SimpleDataAnalyzer:
    """Get the global simple data analyzer instance"""
    return simple_analyzer