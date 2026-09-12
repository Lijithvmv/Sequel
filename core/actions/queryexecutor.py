import asyncio
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from core.actions.querygenerator import QueryGenerator
from core.config import (
    PERMANENT_STORAGE,
    SHARED_VOLUME_STORAGE,
    SQL_ROW_LIMIT,
    Config,
    single_table_description_path,
)
from core.logger import get_logger, log_performance
from core.utils.sqlfilter import add_sql_filter

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Set Logger
logger = get_logger(__name__)
path = os.path.dirname(__file__)

# For local query executor testing
server_hostname = os.getenv("DB-WORKSPACE-URL")
http_path = os.getenv("DB-WAREHOUSE")
sp_client_secret = os.getenv("SP-CLIENT-SECRET")

single_table_descriptions_file = PERMANENT_STORAGE + single_table_description_path
query_generator_prompt = os.path.join(PERMANENT_STORAGE , 'prompts', 'QueryGeneratorPrompt.txt')


class QueryExecutor(object):

    def __init__(self):
        # Parse DATABASE_URL from Config
        db_url = Config.DATABASE_URL
        parsed = urlparse(db_url)
        
        self.db_type = parsed.scheme.split('+')[0]  # mysql, postgresql, etc.
        self.host = parsed.hostname
        self.dbname = parsed.path.lstrip("/")
        self.user = parsed.username
        self.password = parsed.password
        self.connection = None
        self.logger = get_logger(__name__)
        
        # Set default port based on database type
        if self.db_type == 'mysql':
            self.port = parsed.port or 3306
        elif self.db_type == 'postgresql':
            self.port = parsed.port or 5432
        else:
            self.port = parsed.port or 3306

    def connect(self) -> None:
        
        if self.db_type == 'mysql':
            if not MYSQL_AVAILABLE:
                raise ImportError("MySQL connector not available. Please install: pip install mysql-connector-python")
            
            self.logger.info("Establishing connection to the MySQL database")
            self.logger.info(f"Connection params: host={self.host}, port={self.port}, db={self.dbname}, user={self.user}")
            try:
                # Close existing connection if any
                if self.connection:
                    try:
                        if self.connection.is_connected():
                            self.connection.close()
                            self.logger.info("Closed existing connection")
                    except:
                        pass
                
                self.logger.info("Creating new MySQL connection...")
                # Use IP address to avoid DNS lookup issues
                host_ip = "127.0.0.1" if self.host == "localhost" else self.host
                
                self.connection = mysql.connector.connect(
                    host=host_ip,
                    port=self.port,
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    connection_timeout=5,
                    autocommit=True,
                    auth_plugin='mysql_native_password',
                    use_pure=True
                )
                self.logger.info("MySQL connection established successfully")
                
                # Test the connection
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                self.logger.info("Connection test successful")
            except Exception as e:
                self.logger.error(f"Failed to connect to MySQL: {e}")
                raise
                
        elif self.db_type == 'postgresql':
            if not POSTGRES_AVAILABLE:
                raise ImportError("PostgreSQL connector not available. Please install: pip install psycopg2-binary")
            
            self.logger.info("Establishing connection to the PostgreSQL database")
            try:
                self.connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password
                )
            except Exception as e:
                self.logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    @staticmethod 
    def clean_special_characters(df: pd.DataFrame) -> pd.DataFrame:

        return df.applymap(lambda x: re.sub(r'[^a-zA-Z0-9]', '_', x) if isinstance(x, str) else x)

    @staticmethod
    def prepare_query(query: str) -> str:

        single_line_query = query.replace('\n', ' ')
        return re.sub(' +', ' ', single_line_query).strip()
        
    def execute_single_query(self, final_query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Optional[Exception]]:

        try:
            self.logger.info(f"SQL TO EXECUTE: {final_query}")
            self.logger.info("Creating cursor...")
            with self.connection.cursor() as cursor:
                self.logger.info("Setting query timeout...")
                cursor.execute("SET SESSION max_execution_time = 30000")
                self.logger.info("Executing main query...")
                cursor.execute(final_query, params)
                self.logger.info("Fetching results...")
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                self.logger.info("Creating DataFrame...")
            df = pd.DataFrame(result, columns=columns)
            self.logger.info(f"QUERY RESULT (first 5 rows):\n{df.head()}")
            self.logger.info("Query executed successfully")
            return df, None
        except Exception as e:
            self.logger.error("Execution failed: %s", str(e))
            return pd.DataFrame(), e
    
    def close_connection(self):
        """Close the database connection"""
        try:
            if self.connection and self.connection.is_connected():
                self.connection.close()
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")

    @staticmethod
    def is_date_string(s: str) -> bool:

        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{4}/\d{2}/\d{2}$'   # YYYY/MM/DD
        ]
        for pattern in date_patterns:
            if re.match(pattern, s):
                return True
        return False
    
    @staticmethod
    def simplify_date_columns(df: pd.DataFrame) -> pd.DataFrame:

        df_new = df.copy()
        preserve_patterns = {'year', 'date', 'code', 'upc', 'sku', 'site', 'store'}
        
        for col in df_new.columns:
            if any(pattern in col.lower() for pattern in preserve_patterns):
                continue  # Skip columns with preserve patterns
            if pd.api.types.is_datetime64_any_dtype(df_new[col]):
                df_new[col] = df_new[col].dt.strftime('%Y-%m-%d')
            elif df_new[col].dtype == object:
                # Apply the is_date_string function to filter valid date strings
                valid_dates = df_new[col].apply(lambda x: pd.to_datetime(x, errors='coerce', utc=True) if QueryExecutor.is_date_string(str(x)) else pd.NaT)
                if df_new[col].notnull().sum() > 0 and (valid_dates.notnull().sum() ==
                                                        df_new[col].notnull().sum()): 
                    df_new[col] = valid_dates.dt.strftime('%Y-%m-%d')
                
        return df_new     

    def save_results(self,
                    df: pd.DataFrame,
                    title: str,
                    session_id: str,
                    date_format: str = "%Y-%m-%d_%H-%M-%S_%f",
                    float_format: str = "%.2f") -> str:

        timestamp = datetime.now().strftime(date_format)
        file_path = SHARED_VOLUME_STORAGE + f"/dependency/{session_id}/result_{timestamp}.csv"
        df = self.clean_special_characters(df)
        df = QueryExecutor.simplify_date_columns(df)
        df.iloc[:SQL_ROW_LIMIT].to_csv(file_path, float_format=float_format, index=False)
        self.logger.info("Results for '%s' saved to CSV", title)

        return file_path

    def regenerate_query(
        self, title: str, final_query: str, selected_tables: list[str], error: Exception, filter_value: str
    ) -> str:

        try:
            query_generator = QueryGenerator(
                single_table_descriptions_file=single_table_descriptions_file,
                query_generator_prompt=query_generator_prompt,
            )
            regenerated_query = query_generator.generate_query(
                merged_prompt=final_query,
                selected_tables=selected_tables,
                error=str(error),
            )
            new_query = ""
            for table_queries in regenerated_query[0].values():
                for sub_query in table_queries.values():
                    new_query += " " + sub_query
            new_query = new_query.strip()
            
            filtered_query_dict = add_sql_filter({title: new_query}, filter_value)  
            return filtered_query_dict[title]
        except Exception as e:
            self.logger.error("Query regeneration failed with: %s", str(e))
            return final_query
        
    def process_query(
        self,
        title: str, 
        query: str,
        selected_tables: list[str],
        params: Optional[Dict[str, Any]],
        max_retry: int,
        session_id: str,
        filter_value: str
    ) -> Tuple[bool, bool, Optional[str]]:

        final_query = self.prepare_query(query)
        retry = 0 
        while retry < max_retry:
            
            df, error = self.execute_single_query(final_query, params)
            if error is None and not df.empty and df.count().sum() > 0:
                file_path = self.save_results(
                    df=df,
                    title=title,
                    session_id=session_id,
                )
                return True, True, file_path
            elif error is None:
                self.logger.info("No data returned for query: %s", title)
                return True, False, None 

            retry += 1
            if retry < max_retry:
                self.logger.info("Retrying query '%s'", title)
                final_query = self.regenerate_query(title, final_query, selected_tables, error, filter_value)
            else:
                self.logger.error("Max retries reached for query: %s", title)
                            
        return False, False, None       

    @log_performance
    async def execute_query_with_retry(
        self,
        queries: Dict[str, str],
        session_id: str,
        selected_tables: list[str],
        filter_value: str,
        params: Optional[Dict[str, Any]] = None,
        max_retry: int = 3
    ) -> Tuple[Dict[str, str], bool, bool, list[str]]:

        self.logger.info("Starting execution of SQL queries")
        if not self.connection:
            self.connect()

        tasks = [
            asyncio.to_thread(
                self.process_query, title, query, selected_tables, params, max_retry, session_id, filter_value
            )
            for title, query in queries.items()
        ]
        
        results = await asyncio.gather(*tasks)

        result_files = {}
        overall_success_flag = True
        data_returned_flag = False 
        queries_missing_data = []
        
        for title, (query_success, data_returned, file_path) in zip(queries.keys(), results):
            
            if query_success and not data_returned:
                queries_missing_data.append(title)      
            if data_returned and file_path is not None:
                result_files[title] = file_path
                data_returned_flag = True 
            if not query_success:
                overall_success_flag = False 
                
        return result_files, overall_success_flag, data_returned_flag, queries_missing_data


    def close(self) -> None:

        self.logger.info("Closing the database connection")
        if self.connection:
            self.connection.close()
            self.connection = None