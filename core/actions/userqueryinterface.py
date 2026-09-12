import os
import re
from typing import List, Optional

from core.actions.chathistory import get_chat_history_manager
from core.connectors.LLMClient import LLMClient
from core.logger import get_logger

# Set Logger
logger = get_logger(__name__)
path = os.path.dirname(__file__)

class UserQueryInterface:
    """
    Enhanced User Query Interface that handles query merging with chat history context.
    """
    
    def __init__(self, llm_client: Optional[object] = None, session_id: Optional[str] = None):
        self.llm_client = llm_client or LLMClient()
        self.session_id = session_id
        self.chat_history_manager = get_chat_history_manager()
        
        # Keywords that indicate follow-up questions
        self.followup_indicators = [
            'also', 'additionally', 'moreover', 'furthermore', 'and what about',
            'what about', 'how about', 'similar', 'like that', 'same way',
            'too', 'as well', 'besides', 'in addition', 'plus',
            'it', 'this', 'that', 'these', 'those', 'them', 'they'
        ]
        
        # Keywords that indicate context references
        self.context_indicators = [
            'above', 'previous', 'earlier', 'before', 'last', 'recent',
            'mentioned', 'said', 'discussed', 'shown', 'listed'
        ]

    def fetch_and_merge_chat(self, chat_history, user_query=None, session_id=None):
        """
        Enhanced method to merge user query with chat history context.
        
        Args:
            chat_history: Previous chat history (pandas DataFrame or list)
            user_query: Current user query
            session_id: Session identifier
            
        Returns:
            str: Merged/enhanced user query
        """
        if not user_query:
            return user_query or "No query provided"
        
        # Use session_id from parameter or instance
        current_session_id = session_id or self.session_id
        if not current_session_id:
            logger.warning("No session ID provided, cannot merge with chat history")
            return user_query
        
        try:
            # Get recent chat context
            recent_context = self.chat_history_manager.get_recent_context(
                current_session_id, context_limit=3
            )
            
            if not recent_context:
                logger.info("No recent context found, returning original query")
                return user_query
            
            # Check if current query needs context merging
            if self._is_followup_query(user_query):
                merged_query = self._merge_with_context(user_query, recent_context)
                logger.info(f"Merged follow-up query. Original: '{user_query[:50]}...', "
                           f"Merged: '{merged_query[:50]}...'")
                return merged_query
            else:
                logger.info("Query appears to be self-contained, no merging needed")
                return user_query
                
        except Exception as e:
            logger.error(f"Error in fetch_and_merge_chat: {e}")
            return user_query
    
    def _is_followup_query(self, query: str) -> bool:
        """
        Determine if a query is likely a follow-up that needs context.
        
        Args:
            query: User query to analyze
            
        Returns:
            bool: True if query appears to be a follow-up
        """
        query_lower = query.lower()
        
        # Check for follow-up indicators
        has_followup_words = any(indicator in query_lower for indicator in self.followup_indicators)
        
        # Check for context references
        has_context_refs = any(indicator in query_lower for indicator in self.context_indicators)
        
        # Check for pronouns without clear referents
        has_ambiguous_pronouns = bool(re.search(r'\b(it|this|that|these|those|them|they)\b', query_lower))
        
        # Check for incomplete questions (very short queries)
        is_very_short = len(query.split()) < 4 and '?' in query
        
        return has_followup_words or has_context_refs or (has_ambiguous_pronouns and len(query.split()) < 10) or is_very_short
    
    def _merge_with_context(self, current_query: str, recent_context: str) -> str:
        """
        Merge current query with recent context using LLM if available.
        
        Args:
            current_query: The current user query
            recent_context: Recent chat context
            
        Returns:
            str: Merged query with context
        """
        try:
            if self.llm_client and hasattr(self.llm_client, 'send_sync_request'):
                # Use LLM to intelligently merge context
                return self._llm_merge_context(current_query, recent_context)
            else:
                # Fallback to rule-based merging
                return self._rule_based_merge(current_query, recent_context)
        except Exception as e:
            logger.error(f"Error in context merging: {e}")
            return self._rule_based_merge(current_query, recent_context)
    
    def _llm_merge_context(self, current_query: str, recent_context: str) -> str:
        """
        Use LLM to merge query with context intelligently.
        """
        merge_prompt = f"""
Given the recent conversation context and a new user query, create a comprehensive query that includes necessary context for understanding.

Recent Context:
{recent_context}

New Query: {current_query}

Instructions:
1. If the new query refers to previous information (using words like "it", "this", "also", "same"), incorporate that context
2. If the new query is self-contained, return it as is
3. Create a single, clear query that can be understood without the conversation history
4. Keep it concise but complete

Merged Query:"""

        try:
            response = self.llm_client.send_sync_request(merge_prompt)
            # Extract the merged query from response
            merged = response.strip()
            
            # Basic validation - if response is too long or doesn't make sense, fallback
            if len(merged) > len(current_query) * 3 or not merged:
                return self._rule_based_merge(current_query, recent_context)
            
            return merged
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            return self._rule_based_merge(current_query, recent_context)
    
    def _rule_based_merge(self, current_query: str, recent_context: str) -> str:
        """
        Rule-based context merging as fallback.
        
        Args:
            current_query: Current user query
            recent_context: Recent context string
            
        Returns:
            str: Merged query
        """
        try:
            # Extract previous queries and tables from context
            context_lines = recent_context.split('\n')
            previous_queries = []
            previous_tables = []
            
            for line in context_lines:
                if line.startswith('Previous query:'):
                    prev_query = line.replace('Previous query:', '').strip()
                    if prev_query:
                        previous_queries.append(prev_query)
                elif line.startswith('Tables used:'):
                    tables = line.replace('Tables used:', '').strip()
                    if tables:
                        previous_tables.extend([t.strip() for t in tables.split(',')])
            
            current_lower = current_query.lower()
            
            # Simple rule-based merging
            if any(word in current_lower for word in ['also', 'additionally', 'too', 'as well']):
                if previous_queries:
                    merged = f"{previous_queries[-1]} and {current_query.lower()}"
                    return merged
            
            elif any(pronoun in current_lower for pronoun in ['it', 'this', 'that', 'them', 'they']):
                if previous_queries:
                    # Try to replace pronouns with context
                    merged = current_query
                    if previous_tables:
                        # Replace "it" with the most recent table context
                        merged = re.sub(r'\bit\b', f"the {previous_tables[-1]} data", merged, flags=re.IGNORECASE)
                    return merged
            
            # If no specific merge pattern, just add context as prefix
            if previous_queries and len(current_query.split()) < 5:
                return f"Based on the previous query about '{previous_queries[-1][:50]}...', {current_query}"
            
            return current_query
            
        except Exception as e:
            logger.error(f"Rule-based merge failed: {e}")
            return current_query
    
    def store_query_context(self, session_id: str, user_query: str, 
                           tables_referenced: List[str] = None,
                           generated_sql: str = None, query_result = None):
        """
        Store query context for future reference.
        
        Args:
            session_id: Session identifier
            user_query: The user's query
            tables_referenced: Tables that were used
            generated_sql: SQL that was generated
            query_result: Result of the query execution
        """
        try:
            chat_entry = self.chat_history_manager.create_chat_entry(
                user_query=user_query,
                tables_referenced=tables_referenced,
                generated_sql=generated_sql,
                query_result=query_result
            )
            
            self.chat_history_manager.store_chat_entry(session_id, chat_entry)
            logger.info(f"Stored query context for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store query context: {e}")


# Backward compatibility function
def process_user_query(session_id: str, user_query: str, top_chats) -> str:
    """
    Process user query with chat history context - maintains backward compatibility.
    
    Args:
        session_id: Session identifier
        user_query: Current user query
        top_chats: Previous chat history (pandas DataFrame)
        
    Returns:
        str: Processed/merged query
    """
    try:
        interface = UserQueryInterface(session_id=session_id)
        return interface.fetch_and_merge_chat(top_chats, user_query, session_id)
    except Exception as e:
        logger.error(f"Error in process_user_query: {e}")
        return user_query


