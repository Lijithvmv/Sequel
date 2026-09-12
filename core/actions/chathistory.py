import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import get_logger

logger = get_logger(__name__)


class ChatHistoryManager:
    """
    Manages chat history for sessions, storing user queries, table references, and generated SQL.
    """
    
    def __init__(self, base_path: str = "./sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def get_session_path(self, session_id: str) -> Path:
        """Get the path for a specific session."""
        session_path = self.base_path / session_id
        session_path.mkdir(exist_ok=True)
        return session_path
    
    def get_chat_history_file(self, session_id: str) -> Path:
        """Get the chat history file path for a session."""
        return self.get_session_path(session_id) / "chat_history.json"
    
    def store_chat_entry(self, session_id: str, chat_entry: Dict[str, Any]) -> bool:
        """
        Store a chat entry in the session's history.
        
        Args:
            session_id: The session identifier
            chat_entry: Dictionary containing chat information
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            chat_file = self.get_chat_history_file(session_id)
            
            # Load existing history or create new
            chat_history = []
            if chat_file.exists():
                try:
                    with open(chat_file, 'r', encoding='utf-8') as f:
                        chat_history = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning(f"Could not load existing chat history for session {session_id}, starting fresh")
                    chat_history = []
            
            # Add timestamp if not present
            if 'timestamp' not in chat_entry:
                chat_entry['timestamp'] = datetime.now().isoformat()
            
            # Add the new entry
            chat_history.append(chat_entry)
            
            # Save back to file
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stored chat entry for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chat entry for session {session_id}: {e}")
            return False
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session.
        
        Args:
            session_id: The session identifier
            limit: Maximum number of entries to return (most recent first)
            
        Returns:
            List of chat entries
        """
        try:
            chat_file = self.get_chat_history_file(session_id)
            
            if not chat_file.exists():
                return []
            
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
            
            # Sort by timestamp (most recent first)
            chat_history = sorted(chat_history, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            if limit:
                chat_history = chat_history[:limit]
            
            return chat_history
            
        except Exception as e:
            logger.error(f"Failed to retrieve chat history for session {session_id}: {e}")
            return []
    
    def get_recent_context(self, session_id: str, context_limit: int = 5) -> str:
        """
        Get recent chat context as a formatted string for LLM processing.
        
        Args:
            session_id: The session identifier
            context_limit: Number of recent entries to include
            
        Returns:
            Formatted context string
        """
        try:
            recent_chats = self.get_chat_history(session_id, limit=context_limit)
            
            if not recent_chats:
                return ""
            
            context_parts = []
            for chat in reversed(recent_chats):  # Reverse to get chronological order
                user_query = chat.get('user_query', '')
                tables_used = chat.get('tables_referenced', [])
                sql_generated = chat.get('generated_sql', '')
                
                if user_query:
                    context_parts.append(f"Previous query: {user_query}")
                    if tables_used:
                        context_parts.append(f"Tables used: {', '.join(tables_used)}")
                    if sql_generated:
                        # Only show first line of SQL for brevity
                        sql_preview = sql_generated.split('\n')[0][:100]
                        context_parts.append(f"SQL generated: {sql_preview}...")
                    context_parts.append("---")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get recent context for session {session_id}: {e}")
            return ""
    
    def create_chat_entry(self, user_query: str, tables_referenced: List[str] = None, 
                         generated_sql: str = None, query_result: Any = None,
                         additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a standardized chat entry dictionary.
        
        Args:
            user_query: The user's natural language query
            tables_referenced: List of database tables that were referenced
            generated_sql: The SQL query that was generated
            query_result: The result of executing the SQL query
            additional_metadata: Any additional metadata to store
            
        Returns:
            Standardized chat entry dictionary
        """
        # Store full answer_table for follow-up queries
        metadata = additional_metadata or {}
        if query_result and isinstance(query_result, list) and len(query_result) > 1:
            metadata['answer_table'] = query_result
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'tables_referenced': tables_referenced or [],
            'generated_sql': generated_sql or '',
            'query_result_summary': self._summarize_result(query_result),
            'metadata': metadata
        }
        
        return entry
    
    def _summarize_result(self, query_result: Any) -> str:
        """
        Create a summary of the query result for storage.
        
        Args:
            query_result: The result to summarize
            
        Returns:
            String summary of the result
        """
        if query_result is None:
            return "No result"
        
        try:
            if isinstance(query_result, list) and len(query_result) > 0:
                if isinstance(query_result[0], list):
                    # Tabular data format [[headers], [row1], [row2], ...]
                    num_rows = len(query_result) - 1  # Subtract header row
                    return f"{num_rows} rows returned"
                else:
                    return f"{len(query_result)} items returned"
            elif hasattr(query_result, '__len__'):
                return f"{len(query_result)} items returned"
            else:
                return str(query_result)[:100]  # Truncate long results
        except Exception:
            return "Result summary unavailable"
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Clean up chat history files older than specified days.
        
        Args:
            days_old: Remove sessions older than this many days
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            removed_count = 0
            for session_dir in self.base_path.iterdir():
                if session_dir.is_dir():
                    # Check modification time of the directory
                    if session_dir.stat().st_mtime < cutoff_time:
                        try:
                            # Remove the entire session directory
                            import shutil
                            shutil.rmtree(session_dir)
                            removed_count += 1
                            logger.info(f"Removed old session directory: {session_dir.name}")
                        except Exception as e:
                            logger.error(f"Failed to remove session directory {session_dir.name}: {e}")
            
            logger.info(f"Cleanup complete. Removed {removed_count} old session directories.")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")


# Global instance for easy access
_chat_history_manager = None

def get_chat_history_manager(base_path: str = "./sessions") -> ChatHistoryManager:
    """
    Get the global ChatHistoryManager instance.
    
    Args:
        base_path: Base path for session storage
        
    Returns:
        ChatHistoryManager instance
    """
    global _chat_history_manager
    if _chat_history_manager is None:
        _chat_history_manager = ChatHistoryManager(base_path)
    return _chat_history_manager