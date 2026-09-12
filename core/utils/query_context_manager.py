"""
Enhanced Query Context Manager for tracking query-result relationships and follow-up detection.
This module provides intelligent follow-up question detection by correlating queries with their results.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Represents the context of a query and its results"""
    query_id: str
    session_id: str
    original_query: str
    processed_query: str
    tables_referenced: List[str]
    generated_sql: str
    result_summary: Dict[str, Any]
    column_names: List[str]
    row_count: int
    timestamp: str
    query_type: str  # 'aggregation', 'filter', 'list', 'join', etc.
    key_entities: List[str]  # extracted entities like 'employees', 'salary', 'department'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryContext':
        return cls(**data)

class QueryContextManager:
    """Manages query contexts and detects follow-up relationships"""
    
    def __init__(self, base_path: str = "./sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # In-memory cache for recent contexts (last 24 hours)
        self.recent_contexts: Dict[str, List[QueryContext]] = {}
        self.context_expiry_hours = 24
        
        # Keywords for different query types
        self.query_type_keywords = {
            'filter': ['where', 'above', 'below', 'greater', 'less', 'equal', '>', '<', '=', 'filter'],
            'aggregation': ['count', 'sum', 'average', 'mean', 'max', 'min', 'total', 'group by'],
            'list': ['show', 'list', 'display', 'give me', 'get', 'find', 'view'],
            'sort': ['sort', 'order', 'top', 'bottom', 'highest', 'lowest'],
            'join': ['join', 'with', 'and', 'along with', 'including']
        }
        
        # Follow-up indicators
        self.followup_patterns = [
            # Direct follow-ups
            r'\b(now|then|also|additionally|furthermore|moreover)\b',
            r'\b(those|these|them|that|it)\b',
            r'\bfrom (the|that|this) (result|data|list|table)\b',
            
            # Filtering follow-ups
            r'\b(above|below|greater|less|more|fewer) than \d+\b',
            r'\bwith (salary|income|pay|wage|amount)\b',
            r'\bin (the|that) department\b',
            
            # Sorting/limiting follow-ups
            r'\btop \d+\b',
            r'\bbottom \d+\b',
            r'\bfirst \d+\b',
            r'\blast \d+\b',
            r'\bsort(ed)? by\b',
            
            # Aggregation follow-ups
            r'\baverage (of|for)\b',
            r'\btotal (of|for)\b',
            r'\bcount (of|for)\b',
            r'\bsum (of|for)\b'
        ]
        
        self.compiled_followup_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.followup_patterns]
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract key entities from a query"""
        query_lower = query.lower()
        entities = []
        
        # Common business entities
        business_entities = [
            'employee', 'employees', 'staff', 'worker', 'workers',
            'salary', 'income', 'pay', 'wage', 'compensation',
            'department', 'dept', 'division', 'team',
            'product', 'products', 'item', 'items',
            'customer', 'customers', 'client', 'clients',
            'order', 'orders', 'sale', 'sales',
            'revenue', 'profit', 'cost', 'price'
        ]
        
        for entity in business_entities:
            if entity in query_lower:
                entities.append(entity)
        
        return list(set(entities))  # Remove duplicates
    
    def determine_query_type(self, query: str) -> str:
        """Determine the type of query based on keywords"""
        query_lower = query.lower()
        
        for query_type, keywords in self.query_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        
        return 'general'
    
    def create_context(self, session_id: str, original_query: str, processed_query: str,
                      tables_referenced: List[str], generated_sql: str, 
                      result_df: pd.DataFrame = None, result_table: List[List] = None) -> QueryContext:
        """Create a query context from execution results"""
        
        # Generate unique query ID
        query_id = f"{session_id}_{int(datetime.now().timestamp() * 1000)}"
        
        # Extract result information
        if result_df is not None:
            column_names = result_df.columns.tolist()
            row_count = len(result_df)
            # Create summary of numeric columns
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            result_summary = {
                'numeric_columns': numeric_cols.tolist(),
                'categorical_columns': result_df.select_dtypes(include=['object']).columns.tolist(),
                'total_columns': len(result_df.columns),
                'sample_values': {}
            }
            # Add sample values for key columns
            for col in result_df.columns[:5]:  # First 5 columns
                if len(result_df) > 0:
                    result_summary['sample_values'][col] = str(result_df[col].iloc[0])
        elif result_table and len(result_table) > 0:
            column_names = result_table[0] if result_table else []
            row_count = len(result_table) - 1 if len(result_table) > 1 else 0
            result_summary = {
                'total_columns': len(column_names),
                'sample_row': result_table[1] if len(result_table) > 1 else []
            }
        else:
            column_names = []
            row_count = 0
            result_summary = {}
        
        context = QueryContext(
            query_id=query_id,
            session_id=session_id,
            original_query=original_query,
            processed_query=processed_query,
            tables_referenced=tables_referenced,
            generated_sql=generated_sql,
            result_summary=result_summary,
            column_names=column_names,
            row_count=row_count,
            timestamp=datetime.now().isoformat(),
            query_type=self.determine_query_type(original_query),
            key_entities=self.extract_entities(original_query)
        )
        
        return context
    
    def store_context(self, context: QueryContext) -> bool:
        """Store a query context both in memory and on disk"""
        try:
            # Store in memory cache
            if context.session_id not in self.recent_contexts:
                self.recent_contexts[context.session_id] = []
            
            self.recent_contexts[context.session_id].append(context)
            
            # Keep only recent contexts (last 10 queries per session)
            self.recent_contexts[context.session_id] = self.recent_contexts[context.session_id][-10:]
            
            # Store on disk
            session_path = self.base_path / context.session_id
            session_path.mkdir(exist_ok=True)
            
            context_file = session_path / "query_contexts.jsonl"
            with open(context_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(context.to_dict()) + '\n')
            
            logger.info(f"Stored context for query: {context.original_query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            return False
    
    def get_recent_context(self, session_id: str, limit: int = 1) -> List[QueryContext]:
        """Get recent query contexts for a session"""
        if session_id in self.recent_contexts:
            return self.recent_contexts[session_id][-limit:]
        
        # Load from disk if not in memory
        try:
            session_path = self.base_path / session_id
            context_file = session_path / "query_contexts.jsonl"
            
            if not context_file.exists():
                return []
            
            contexts = []
            with open(context_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        context_data = json.loads(line.strip())
                        context = QueryContext.from_dict(context_data)
                        
                        # Only include recent contexts (last 24 hours)
                        context_time = datetime.fromisoformat(context.timestamp)
                        if datetime.now() - context_time < timedelta(hours=self.context_expiry_hours):
                            contexts.append(context)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping invalid context line: {e}")
                        continue
            
            # Store in memory for future use
            self.recent_contexts[session_id] = contexts[-10:]  # Keep last 10
            
            return contexts[-limit:]
            
        except Exception as e:
            logger.error(f"Failed to load contexts from disk: {e}")
            return []
    
    def is_followup_query(self, current_query: str, session_id: str) -> Tuple[bool, Optional[QueryContext], float]:
        """
        Determine if current query is a follow-up to the most recent query.
        
        Returns:
            (is_followup, relevant_context, confidence_score)
        """
        recent_contexts = self.get_recent_context(session_id, limit=1)
        
        if not recent_contexts:
            return False, None, 0.0
        
        last_context = recent_contexts[0]
        current_query_lower = current_query.lower()
        
        confidence_score = 0.0
        
        # 1. Check for explicit follow-up patterns
        pattern_matches = sum(1 for pattern in self.compiled_followup_patterns 
                            if pattern.search(current_query))
        if pattern_matches > 0:
            confidence_score += 0.3 * min(pattern_matches, 3)  # Cap at 0.9
        
        # 2. Check for entity overlap
        current_entities = self.extract_entities(current_query)
        entity_overlap = len(set(current_entities) & set(last_context.key_entities))
        if entity_overlap > 0 and last_context.key_entities:
            overlap_ratio = entity_overlap / len(last_context.key_entities)
            confidence_score += 0.4 * overlap_ratio
        
        # 3. Check for column name references
        column_mentions = sum(1 for col in last_context.column_names 
                            if col.lower() in current_query_lower)
        if column_mentions > 0:
            confidence_score += 0.2 * min(column_mentions / len(last_context.column_names), 1.0)
        
        # 4. Check for result-specific language
        result_keywords = ['result', 'data', 'list', 'table', 'those', 'these', 'them', 'that']
        result_mentions = sum(1 for keyword in result_keywords if keyword in current_query_lower)
        if result_mentions > 0:
            confidence_score += 0.1 * min(result_mentions, 2)
        
        # 5. Time-based relevance (recent queries are more likely to be follow-ups)
        time_diff = datetime.now() - datetime.fromisoformat(last_context.timestamp)
        if time_diff < timedelta(minutes=5):
            confidence_score += 0.1
        elif time_diff < timedelta(minutes=30):
            confidence_score += 0.05
        
        # 6. Query type compatibility
        current_type = self.determine_query_type(current_query)
        if self._are_compatible_query_types(last_context.query_type, current_type):
            confidence_score += 0.1
        
        # Normalize confidence score
        confidence_score = min(confidence_score, 1.0)
        
        # Decision threshold
        is_followup = confidence_score >= 0.5
        
        logger.info(f"Follow-up analysis: '{current_query[:50]}...' -> "
                   f"Confidence: {confidence_score:.2f}, Is follow-up: {is_followup}")
        
        return is_followup, last_context if is_followup else None, confidence_score
    
    def _are_compatible_query_types(self, last_type: str, current_type: str) -> bool:
        """Check if two query types are compatible for follow-up"""
        # Most filter operations can follow list operations
        if last_type == 'list' and current_type in ['filter', 'sort', 'aggregation']:
            return True
        
        # Aggregations can follow filters
        if last_type == 'filter' and current_type in ['aggregation', 'sort']:
            return True
        
        # General follow-ups
        if current_type in ['filter', 'sort', 'aggregation']:
            return True
        
        return False
    
    def get_context_summary(self, context: QueryContext) -> str:
        """Get a human-readable summary of a query context"""
        return (f"Query: {context.original_query}\n"
                f"Tables: {', '.join(context.tables_referenced)}\n" 
                f"Results: {context.row_count} rows, {len(context.column_names)} columns\n"
                f"Type: {context.query_type}\n"
                f"Entities: {', '.join(context.key_entities)}")
    
    def cleanup_old_contexts(self, days_old: int = 7):
        """Clean up old context files"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            
            for session_dir in self.base_path.iterdir():
                if session_dir.is_dir():
                    context_file = session_dir / "query_contexts.jsonl"
                    if context_file.exists():
                        # Read and filter recent contexts
                        recent_contexts = []
                        with open(context_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    context_data = json.loads(line.strip())
                                    context_time = datetime.fromisoformat(context_data['timestamp'])
                                    if context_time > cutoff_time:
                                        recent_contexts.append(line.strip())
                                except (json.JSONDecodeError, KeyError):
                                    continue
                        
                        # Rewrite file with only recent contexts
                        if recent_contexts:
                            with open(context_file, 'w', encoding='utf-8') as f:
                                for context_line in recent_contexts:
                                    f.write(context_line + '\n')
                        else:
                            context_file.unlink()  # Remove empty file
            
            logger.info(f"Cleaned up contexts older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old contexts: {e}")

# Global instance
_query_context_manager = None

def get_query_context_manager(base_path: str = "./sessions") -> QueryContextManager:
    """Get the global QueryContextManager instance"""
    global _query_context_manager
    if _query_context_manager is None:
        _query_context_manager = QueryContextManager(base_path)
    return _query_context_manager