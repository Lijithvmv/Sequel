import os
import re
import time

from core.logger import get_logger, log_performance
from core.utils.utils import extract_nested_dict

logger = get_logger(__name__)
path = os.path.dirname(__file__)


class QueryGenerator(object):


    def __init__(self,
                 llm_client,
                 single_table_descriptions_file: str,
                 query_generator_prompt: str = "prompts/query_generator.txt",
                 max_retries: int = 3,
                 retry_wait_time: int = 2
                 ):

        self.logger = get_logger(__name__)
        self.query_generator_prompt = query_generator_prompt
        self.single_table_descriptions_file = single_table_descriptions_file
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.retry_wait_time = retry_wait_time 
        self.attempt = 0
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:

        self.logger.info('Load the system Prompt')
        try:
            with open(self.query_generator_prompt, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error("Failed to load Query_generator_instruction_file: %s", e)
        return ''

    def all_table_desc(self, table):
        # Return a real schema string for each table
        if table == "employees":
            return "Table: employees (employee_id, name, department_id, salary)"
        elif table == "departments":
            return "Table: departments (department_id, department_name, manager_id)"
        # Add relationships if relevant
        return ""

    @staticmethod
    def clean_query(query: str) -> str:

        logger.info('Cleaning up the Generated SQL Query')
        # Replace newlines with a space, then collapse multiple spaces
        cleaned_query = query.replace('\n', ' ')
        cleaned_query = re.sub(' +', ' ', cleaned_query).strip()
        return cleaned_query

    @staticmethod
    def extract_sql_from_llm_response(response):
        # Try to find a code block with SQL
        match = re.search(r"```sql\\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: try to find the first SELECT statement (even if not in a code block)
        match = re.search(r"(SELECT[\s\S]+?;)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: try to get all lines from the first SELECT to the end if no semicolon
        lines = response.splitlines()
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("SELECT"):
                return '\n'.join(lines[i:]).strip()
        # New fallback: try to find any line containing both SELECT and FROM
        for line in lines:
            if "SELECT" in line.upper() and "FROM" in line.upper():
                return line.strip()
        return None

    @log_performance
    def generate_sql_for_table(self, user_query: str, table: str, table_description: str, error: str) -> str:
        prompt = self.prompt_template.format(
            user_query=user_query,
            table_name=table,
            table_description=table_description
        )
        for attempt in range(self.max_retries):
            try:
                self.logger.info('Sending Prompt to LLM')
                llm_result = self.llm_client.send_sync_request(prompt)["choices"][0]["message"]["content"]
                # Print the raw LLM response for debugging
                print(f"RAW LLM RESPONSE for table {table}:\n{llm_result}")
                self.logger.info(f"RAW LLM RESPONSE for table {table}:\n{llm_result}")
                if not llm_result:
                    raise ValueError("Invalid response received from LLM")
                return llm_result
            except Exception as e:
                self.logger.error("Error constructing SQL query: %s", e)
                self.attempt += 1
                if self.attempt < self.max_retries:
                    self.logger.info(f"Retrying... attempt {self.attempt}/{self.max_retries}")
                    time.sleep(self.retry_wait_time)
                else:
                    self.logger.error(f"Max retries reached. Giving up after {self.max_retries} attempts.")
                    raise type(e)(str(e)) from e

    @log_performance
    def generate_sql_queries(self, merged_prompt: str, selected_tables: list, error: str) -> tuple:
        self.logger.info('start Generating Query')
        title_queries = {}
        debug_info = {}
        self.logger.info(f"MERGED PROMPT: {merged_prompt}")
        for table in selected_tables:
            self.logger.info(f"--- Processing table: {table} ---")
            table_descriptions = self.all_table_desc(table)
            title_queries[table] = {}
            prompt = None  # Ensure prompt is always defined
            try:
                table_desc_str = str(table_descriptions)
                prompt = self.prompt_template.format(
                    user_query=merged_prompt,
                    table_name=table,
                    table_description=table_desc_str
                )
                self.logger.info(f"PROMPT SENT TO LLM for table {table}:\n{prompt}")
                response = self.generate_sql_for_table(merged_prompt, table, table_desc_str, error)
                self.logger.info(f"RAW LLM RESPONSE for table {table}:\n{response}")
                debug_info[table] = {"llm_prompt": prompt, "llm_response": response}
                try:
                    parsed = extract_nested_dict(response)
                    num_query = parsed.get("queries", [])
                    self.logger.info(f"PARSED QUERIES for table {table}: {num_query}")
                    debug_info[table]["parsed_queries"] = num_query
                    if not num_query:
                        raise ValueError("No 'queries' key or empty queries")
                    query_len = len(num_query)
                    for q in range(query_len):
                        title = num_query[q]["title"]
                        query = num_query[q]["sql"]
                        title_queries[table][title] = QueryGenerator.clean_query(str(query))
                except Exception as e:
                    # Fallback: try to extract SQL from LLM response
                    self.logger.error(f"Error parsing JSON or missing queries, trying fallback extraction: {e}")
                    sql = self.extract_sql_from_llm_response(response)
                    debug_info[table]["extracted_sql"] = sql
                    self.logger.info(f"EXTRACTED SQL for table {table}: {sql}")
                    if sql:
                        title_queries[table]["LLM SQL"] = QueryGenerator.clean_query(sql)
                        self.logger.info(f"Extracted SQL for table {table}: {sql}")
                    else:
                        self.logger.error(f"Could not extract SQL for table {table}")
            except Exception as e:
                # Fallback: try to extract SQL from the exception message (which may contain the LLM response)
                self.logger.error(f"Error generating query for table {table}, trying fallback extraction: {e}")
                sql = self.extract_sql_from_llm_response(str(e))
                debug_info[table] = {"llm_prompt": prompt, "exception": str(e), "extracted_sql": sql}
                self.logger.info(f"EXTRACTED SQL for table {table}: {sql}")
                if sql:
                    title_queries[table]["LLM SQL"] = QueryGenerator.clean_query(sql)
                    self.logger.info(f"Extracted SQL for table {table}: {sql}")
                else:
                    self.logger.error(f"Could not extract SQL for table {table}")
        self.logger.info(f"generated query for {title_queries}")
        query_generator_flag = 0
        if any(len(queries) > 0 for queries in title_queries.values()) :
            query_generator_flag = 1
            self.logger.info("Query Generator process completed successfully")
        # Return debug_info for UI display (as part of the tuple for now)
        return title_queries, query_generator_flag, debug_info
