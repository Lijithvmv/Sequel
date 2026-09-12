# General
import re
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ValidationError, field_validator

# Local Imports
from core.logger import get_logger

# Set Logger
logger = get_logger(__name__)

RESERVED_WORDS = {
    "select", "from", "where", "join", "and", "or", "on", "as", "in",
    "group", "by", "order", "limit", "having", "union", "intersect", "except"
}

class FilterConfig(BaseModel):
    """ 
    Configuration class for filtering SQL queries based on specified filter value and column settings.
    
    Attributes:
        queries (Dict[str, str]): Dictionary of SQL query strings, keyed by titles.
        filter_value (Union[str, int, float]): Value to use in filtering conditions.
        filter_column (str): Name of the database column for filtering.
        whitelisted_tables (Optional[List[str]]): List of table names exempt from filtering.
    """
    queries: Dict[str, str]
    filter_value: Union[str, int, float] 
    filter_column: str 
    whitelisted_tables: Optional[List[str]] = None 
    
    @field_validator('queries')
    def validate_queries(cls, v: Dict[str, str]) -> Dict[str, str]:
        """ 
        Validate the queries dictionary.
        
        Args:
            v (Dict[str, str]): The queries dictionary
        
        Returns:
            Dict[str, str]: The validated queries dictionary 
          
        Raises:
            ValueError: If the queries are invalid        
        """
        if not isinstance(v, dict):
            raise ValueError(f"Invalid queries type: {type(v)}. Expected dict.")
        if not all(isinstance(k, str) and isinstance(val, str) for k, val in v.items()):
            raise ValueError("Invalid query dictionary: all keys and valuest must be string.") 
        return v   
    
    @field_validator('filter_value')
    def sanitize_filter_value(cls, v: Union[str, int, float]) -> Union[str, int, float]:
        """ 
        Sanitize filter value. For strings, escape single quotes.
        For numeric types, ensure the are valid numbers.
        
        Args:
            v (str): The input filter value.
            
        Returns:
            Union[str, int, float]: Sanitized filter value.
            
        Riases:
            ValueError: If the filter value is not a non-empty string.        
        """
        if isinstance(v, str):
            if not v:
                raise ValueError("String filter_value must be non-empty")
            return v.replace("'", "''")
        elif isinstance(v, (int, float)):
            return v
        else:
            raise ValueError("filter_value must be a string, integer, or float")

    @field_validator('filter_column')
    def validate_filter_column(cls, v: str) -> str:
        """ 
        Ensure filter_column is a valid SQL identifier.
        
        Args:
            v (str): The input filter column name.
            
        Returns:
            str: The validated filter column name.
            
        Raises:
            ValueError: If the column name is not a valid SQL identifier.        
        """
        if not re.match(r'^[A-Za-z_]\w*$', v):
            raise ValueError(f"Invalid SQL identifier for filter_column: {v}")
        return v 
    
    @field_validator('whitelisted_tables')
    def validate_whitelisted_tables(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """
        Validate and sanitize the whiteliste_tables field.
        Ensures that each table name is lowercase and non-empty.
        
        Args:
            v (Optional[List[str]]): The list of input table names.
            
        Returns:
            Optional[List[str]]: Sanitized list of lowercase table names
            
        Riases:
            ValueError: If the table name is not a non-empty string.        
        """
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("whitelisted_tables must be a list or None")
        if not all(isinstance(item, str) and item for item in v):
            raise ValueError("Each whitelisted table must be a non-empty string")
        return [item.lower() for item in v]


class SQLQueryFilter(object):
    """
    A class for filtering SQL queries by appending specified WHERE clauses.
    
    This class processes SQL queries to append filtering conditions while respecting whitelisted
    tables. It handles complex SQL queries including UNIONs, subqueries, and CTEs.
    """
    def __init__(
        self,
        config: FilterConfig
    ) -> None:
        """
        Initialize the SQL Query Filter.
        
        Args:
            config: An instance of FilterConfig containing all necessary configurations.
            
        Raises:
            ValueError: If any input parameters are invalid
        """
        try:
            self.config = config
            self.logger = get_logger(__name__)
            self.queries = config.queries
            self.filter_value = config.filter_value
            self.filter_column = config.filter_column 
            self.whitelisted_tables = [table.lower() for table in (config.whitelisted_tables or [])]
        except ValidationError as e:
            logger.info(f"Configuration validation error: {e}")
            raise
             
    @staticmethod    
    def extract_tables(query: str) -> List[str]:
        """ 
        Extract all table names from a SQL query.
        
        Args:
            query: The SQL query to analyze
        
        Returns:
            List of table names found in the query    
        """        
        try:

            clean_query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
            clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)
            clean_query = re.sub(r"'[^']*'", '', clean_query)
            clean_query = re.sub(r'"[^"]*"', '', clean_query)

            clean_query = re.sub(
                r"EXTRACT\s*\(\s*[^)]*FROM[^)]*\)",
                "",
                clean_query,
                flags=re.IGNORECASE | re.DOTALL
            )

            cte_pattern = r'\bWITH\s+([^)]+)\)'
            clean_query = re.sub(cte_pattern, '', clean_query, flags=re.IGNORECASE)

            tables = []
            from_pattern = r'\bFROM\s+([\w."$]+(?:\s*,\s*[\w."$]+)*)'
            tables.extend(re.findall(from_pattern, clean_query, re.IGNORECASE))
            
            join_pattern = r'\bJOIN\s+([\w."$]+)'
            tables.extend(re.findall(join_pattern, clean_query, re.IGNORECASE))

            result = []
            for group in tables:
                for token in group.split(','):
                    token = token.strip().lower().split()[0]
                    table_name = token.split('.')[-1]
                    table_name = table_name.strip('"').strip("'")
                    table_name = re.sub(r'[^\w]','', table_name)
                    result.append(table_name)
                result = [t for t in result if t not in RESERVED_WORDS]
            return list(set(result))            
        except Exception as e:
            logger.info(f"Error extracting tables: {str(e)}, Query: {query}")
            return []
        
    def should_skip_filtering(self, query: str) -> bool:
        """ 
        Decide whether to skip filtering if the query references any whitelisted table or
        if no tables are found at all.
        
        Args:
            query: The SQL query to check.
            
        Returns:
            True if all tables in the query are whitelisted, False otherwise    
        """
        try:
            tables = SQLQueryFilter.extract_tables(query)
            if not tables:
                self.logger.info(f"Invalid SQL - no tables found in query: {query}")
                return True     
            any_whitelisted = any(table in self.whitelisted_tables for table in tables)
            if any_whitelisted:
                self.logger.info(f"Found a whitelisted table, skipping filter. Tables: {tables}")
            return any_whitelisted  
        except Exception as e:
            self.logger.info(f"Error checking whitelist status: {str(e)}, Query: {query}")
            return False 
        
    @staticmethod    
    def has_where_clause(query: str) -> Tuple[bool, Optional[int]]:
        """ 
        Check for existence and position of WHERE clause in query.
        
        Args:
            query: The SQL query to check
            
        Returns:
            Tuple of (has where clause, position of where clause)
            Position will be None if no WHERE clause is found    
        """        
        try:
            pattern = r'\bWHERE\b(?![^()]*\))'
            match = re.search(pattern, query, re.IGNORECASE)
            return (bool(match), match.start() if match else None)
        except Exception as e:
            logger.info(f"Error checking WHERE clause: {str(e)}, Query: {query}")
            return (False, None)
        
    @staticmethod    
    def add_where_to_simple_query(query: str, filter_condition: str) -> Optional[str]:
        """ 
        Add WHERE clause to a query that doesn't have one.
        
        Args:
            query: Original SQL query string
            filter_condition: Filtering condition to add
            
        Returns:
            Modified query string with WHERE clause added,
            or None if processing fails   
        """    
        try:
            for clause in ['GROUP BY', 'ORDER BY', 'LIMIT']:
                pattern = re.compile(fr'\b{clause}\b', re.IGNORECASE)
                match = pattern.search(query)
                if match:
                    return (f"{query[:match.start()]} WHERE {filter_condition} {query[match.start():]}")
                    
            return f"{query} WHERE {filter_condition}"
        except Exception as e:
            logger.info(f"Error adding WHERE to simple query: {str(e)}, Query: {query}")
            return None 
        
    def process_single_query(self, query: str) -> Optional[str]:
        """ 
        Process a single (non-multipart) query.
        
        Args:
            query: SQL query string to process
            
        Returns:
            Modified query string with filtering applied, or None if processing fails    
        """                
        filter_condition = self.construct_filter_condition()
        has_where, where_pos = SQLQueryFilter.has_where_clause(query)
        
        if has_where and where_pos is not None:
            where_clause = query[where_pos:]
            if re.search(re.escape(filter_condition), where_clause, re.IGNORECASE):
                logger.info("Filter condition already present in WHERE claise")
                return query 
            return (f"{query[:where_pos].rstrip()} WHERE {filter_condition} AND {query[where_pos + 5:].lstrip()}") # +5 to skip "WHERE"
            
        return SQLQueryFilter.add_where_to_simple_query(query, filter_condition)
    
    @staticmethod
    def extract_query_parts(query: str) -> List[str]:
        """ 
        Split complex SQL queries on UNION/INTERSECT/EXCEPT so we can filter each piece separately
        
        Args:
            query: The original SQL query.
            
        Returns:
            A list of individual query segments (excluding keywords themselves)
        """
        pattern = r'\b(UNION\s+ALL|UNION|INTERSECT|EXCEPT)\b(?![^()]*\))'
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        return [
            part.strip()
            for part in parts 
            if not re.match(r'^(UNION\s+ALL|UNION|INTERSECT|EXCEPT)$', part.strip(), re.IGNORECASE)
        ]
    
    def handle_multipart_query(self, query: str, query_parts: List[str]) -> Optional[str]:
        """
        Process queries containing UNION, INTERSECT, or EXCEPT.
        
        Args:
            query: Original SQL query string
            query_parts: List of component queries 
        
        Returns:
            Modified query string with filtering conditions added to each part,
            or None if processing fails
        """    
        try:
            modified_parts = []
            for part in query_parts:
                modified_part = self.append_filter_condition(part)
                if modified_part is None:
                    return None 
                modified_parts.append(modified_part)
                
            separators = re.findall(
                r'\b(UNION\s+ALL|UNION|INTERSECT|EXCEPT)\b(?![^()]*\))',
                query,
                flags=re.IGNORECASE
            )    
            result = modified_parts[0]
            for i, sep in enumerate(separators):
                result += f" {sep.upper()} {modified_parts[i+1]}"
            return result 
        except Exception as e:
            self.logger.info(f"Error handling multipart query: {str(e)}, Query: {query}")
            return None 
        
    def process_query_with_rules(self, query: str) -> Optional[str]:
        """ 
        Apply filtering rules to a query
        
        Args:
            query: SQL query string to process
            
        Returns:
            Modified query string with filtering rules applied,
            or None of processing fails    
        """        
        if self.should_skip_filtering(query):
            return query
        
        query_parts = SQLQueryFilter.extract_query_parts(query)
        if len(query_parts) > 1:
            return self.handle_multipart_query(query, query_parts)
        
        return self.process_single_query(query)
    
    def handle_with_query(self, query: str) -> Optional[str]:
        """ 
        Processes queries starting with a WITH clause applying the filter condition to each
        CTE subquery as well as to the main query.
        
        Args:
            query: SQL query string starting with WITH  
        
        Returns:
            Modified SQL query with filtering condition applied, or None if processing fails    
        """
        try:
            pos_match = re.search(r'\)\s*(SELECT\b)', query, flags=re.IGNORECASE | re.DOTALL)
            if not pos_match:
                return self.process_query_with_rules(query)
            
            split_pos = pos_match.start() + 1 
            with_clause = query[:split_pos]
            main_query = query[split_pos:]

            cte_pattern = re.compile(
                r'(\w+\s+AS\s*\()'
                r'(.*?)'
                r'(\)\s*(?:,|$|\n|SELECT))',
                flags=re.IGNORECASE | re.DOTALL
            )
           
            new_with_clause = cte_pattern.sub(self.replace_cte_subquery, with_clause)
            
            cte_names = re.findall(r'\b(\w+)\s+AS\s*\(', with_clause, flags=re.IGNORECASE)
            cte_names = [name.lower() for name in cte_names]
            
            main_tables = SQLQueryFilter.extract_tables(main_query)
            
            if main_tables and all(t in cte_names for t in main_tables):
                new_main_query = main_query
            else:
                new_main_query = self.append_filter_condition(main_query) or main_query
    
            return f"{new_with_clause} {new_main_query}"
        except Exception as e:
            logger.info(f"Error handling WITH query: {str(e)}, Query: {query}")
            return None 
      
    def replace_cte_subquery(self, match: re.Match) -> str:
        """
        Regex replacement method for a single CTE subquery capture.
        
        Args:
            match: Regex match object
         
        Returns:
            str: The rebuilt string containing the (optionally) filtered subquery    
        """  
        cte_intro = match.group(1)
        cte_sql = match.group(2).strip()
        cte_close = match.group(3)
        
        filtered_subquery = self.append_filter_condition(cte_sql)
        if filtered_subquery is None:
            filtered_subquery = cte_sql  
        return f"{cte_intro}{filtered_subquery}{cte_close}"
        
    def replace_joins(self, match: re.Match) -> str: 
        """ 
        Helper function to process subqueries in JOIN claises
        
        Args:
            match: A regex match object capturing CTE name and its subquery 
         
       Returns:
            A string with the CTE name and its filterd subquery     
        """        
        cte_name = match.group(1)
        subquery_with_paren = match.group(2)
        
        if subquery_with_paren.endswith(')'):
            subquery = subquery_with_paren[:-1]
            trailing = ')'
        else: 
            subquery = subquery_with_paren
            trailing = ''    
        filtered_subquery = self.append_filter_condition(subquery)
        if filtered_subquery is None: 
            filtered_subquery = subquery
        return f"{cte_name} ({filtered_subquery}{trailing}"    
    
    def process_join_subqueries(self, query: str) -> str:
        """
        Process subqueries that are part of JOIN claises by filtering them.
        
        Args:
            query: The SQL query string to process.
            
        Returns:
            Modified SQL query with filtered JOIN subqueries.
        """    
        pattern = re.compile(
            r'(?i)(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|JOIN)\s*\(\s*(SELECT[^)]*\))',
            re.DOTALL
        )
        return pattern.sub(self.replace_joins, query)
    
    @staticmethod
    def normalize_table_id_replacer(match: re.Match) -> str:
        """ 
        Helper function to normalize table identifiers in FROM and JOIN clauses
        
        Args:
            match: A regex match object capturing the clause and quoted table identifier
            
        Returns:
            Modified SQL query with normalized table identifiers.    
        """
        clause = match.group(1)
        table = match.group(3)
        normalized = re.sub(r'[^\w]', '', table)
        return f"{clause} {normalized}"
    
    @staticmethod
    def normalize_table_identifiers(query: str) -> str:
        """ 
        Normalize table identifiers in FROM and JOIN claises by removing quotes and disallowed characters.
        
        Args:
            query: The SQL query string to normalize
         
        Returns:
            Modified SQL query with normalized table identifiers.    
        """
        pattern = re.compile(r'(?i)\b(FROM|JOIN)\s+(")([a-z0-9\$]+)(")')
        return pattern.sub(SQLQueryFilter.normalize_table_id_replacer, query)
    
    def append_filter_condition(self, query: str) -> Optional[str]:
        """ 
        Main method to append filtering condition to a query.
        
        Args:
            query: SQL query string to process
            
        Returns:
            Modified query string with filtering condition applied,
            or None if processing fails    
        """
        if query.lstrip().upper().startswith("WITH"):
            processed = self.handle_with_query(query)
        else:    
            processed = self.process_query_with_rules(query)
        if processed is None: 
            return None 
        processed = self.process_join_subqueries(processed)
        processed = SQLQueryFilter.normalize_table_identifiers(processed)
        return processed    
      
    def construct_filter_condition(self) -> str:
        """ 
        Construct the filtering condition based on the filter_value type.
        
        Returns:
            A string representing the filtering condition.
            
        Raise:
            ValueError: If the filter value is not a string or numeric value.    
        """
        if isinstance(self.filter_value, str):
            return f"{self.filter_column} = '{self.filter_value}'"
        elif isinstance(self.filter_value, (float, int)):
            return f"{self.filter_column} = {self.filter_value}"    
        else:
            self.logger.info(f"Unsupported filter_value tpe: {type(self.filter_value)}")
            raise ValueError("Unsupported filter_value type")
    
    def get_filtered_queries(self) -> Dict[str, str]:
        """ 
        Get all queries with filtering conditions applied.
        
        Returns:
            Dictionary mapping query titles to filtered SQL queries.
            Returns empty dictionary if any error occurs during processing
        """        
        try:
            filtered_queries = {}
            for title, query in self.queries.items():
                filtered_query = self.append_filter_condition(query)
                if filtered_query is None: 
                    self.logger.info(f"Failed to process query with title: {title}")
                    continue
                filtered_queries[title] = filtered_query  
            self.logger.info(f"Successfully processed queries: {filtered_queries}")
            return filtered_queries
        except Exception as e:
            self.logger.info(f"Error processing queries: {str(e)}")
            return {}        


def add_sql_filter(queries: dict, filter_value: str, filter_column: str = "EmailAddress"):
    # Dummy implementation for now
    return queries