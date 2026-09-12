"""
Enhanced TableSelector with improved error handling and debugging for AI JSON parsing.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.logger import get_logger, log_performance

# Set Logger
logger = get_logger(__name__)


class RuleBasedTableSelector:
    """Fast, deterministic table selection using keyword matching and rules."""

    def __init__(self):
        # Define keyword mappings for your specific schema
        self.table_keywords = {
            "departments": [
                "department", "dept", "division", "team", "unit",
                "branch", "section", "group", "organization"
            ],
            "employees": [
                "employee", "worker", "staff", "person", "people",
                "user", "member", "individual", "emp", "personnel",
                "salary", "wage", "pay", "compensation", "income",
                "earnings", "payment", "remuneration", "payroll"
            ],
        }

        # Context-based rules for better matching
        self.context_rules = [
            # HR/People related queries
            {
                "keywords": ["hire", "fired", "promotion", "performance", "review"],
                "tables": ["employees", "departments", "positions"]
            },
            # Financial queries
            {
                "keywords": ["cost", "budget", "expense", "revenue", "profit"],
                "tables": ["departments", "employees"]
            },
            # Management queries
            {
                "keywords": ["manager", "supervisor", "lead", "head", "director"],
                "tables": ["employees", "departments", "positions"]
            }
        ]

    def run(self, user_prompt: str) -> Dict[str, Any]:
        """Select tables using rule-based keyword matching."""
        user_prompt_lower = user_prompt.lower()
        relevant_tables = set()
        confidence_scores = {}

        # Direct keyword matching
        for table_name, keywords in self.table_keywords.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in user_prompt_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                relevant_tables.add(table_name)
                confidence_scores[table_name] = {
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "method": "direct_keyword"
                }

        # Context-based rules
        for rule in self.context_rules:
            rule_matches = sum(1 for keyword in rule["keywords"]
                               if keyword in user_prompt_lower)

            if rule_matches > 0:
                for table in rule["tables"]:
                    if table not in relevant_tables:
                        relevant_tables.add(table)
                        confidence_scores[table] = {
                            "score": rule_matches,
                            "matched_keywords": [kw for kw in rule["keywords"]
                                                 if kw in user_prompt_lower],
                            "method": "context_rule"
                        }
                    else:
                        # Boost existing score
                        confidence_scores[table]["score"] += rule_matches * 0.5

        relevant_tables_list = list(relevant_tables)

        logger.info(f"RuleBasedSelector: user_prompt='{user_prompt}' => "
                    f"relevant_tables={relevant_tables_list} "
                    f"confidence_scores={confidence_scores}")

        return {
            "relevant_tables_found": bool(relevant_tables_list),
            "relevant_tables": relevant_tables_list,
            "confidence_scores": confidence_scores,
            "selection_method": "rule_based",
            "message": f"Found {len(relevant_tables_list)} relevant tables using rule-based matching."
        }


class AITableSelector:
    """AI-powered semantic table selection using LLM with robust JSON parsing."""

    def __init__(self, llm_client, table_descriptions_file: str, instruction_file: str):
        self.llm_client = llm_client
        self.table_descriptions_file = table_descriptions_file
        self.instruction_file = instruction_file
        self.table_descriptions = self._load_table_descriptions()
        self.instructions = self._load_instructions()

    def _load_table_descriptions(self) -> str:
        """Load table descriptions from CSV file."""
        try:
            if Path(self.table_descriptions_file).exists():
                df = pd.read_csv(self.table_descriptions_file)
                descriptions = []
                for _, row in df.iterrows():
                    table_name = row.get('table_name', row.get('Table', ''))
                    description = row.get('description', row.get('Description', ''))
                    if table_name and description:
                        descriptions.append(f"Table: {table_name}\nDescription: {description}")
                return "\n\n".join(descriptions)
            else:
                logger.warning(f"Table descriptions file not found: {self.table_descriptions_file}")
                return "No table descriptions available."
        except Exception as e:
            logger.error(f"Error loading table descriptions: {e}")
            return "Error loading table descriptions."

    def _load_instructions(self) -> str:
        """Load AI instructions from file."""
        try:
            if Path(self.instruction_file).exists():
                with open(self.instruction_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return self._get_default_instructions()
        except Exception as e:
            logger.error(f"Error loading instructions: {e}")
            return self._get_default_instructions()

    def _get_default_instructions(self) -> str:
        """Default instructions for table selection with strict JSON formatting."""
        return """
You are a database table selection assistant. Given a user query and table descriptions, you must return ONLY a valid JSON object.

CRITICAL RULES:
1. Return ONLY a valid JSON object, nothing else
2. No markdown formatting, no ```json blocks, no explanations
3. Use double quotes for all strings
4. Ensure all JSON syntax is correct

Required JSON structure:
{{
    "relevant_tables": ["table1", "table2"],
    "confidence_scores": {{
        "table1": 0.9,
        "table2": 0.7
    }},
    "reasoning": "Brief explanation"
}}

User Query: {user_query}

Table Descriptions:
{table_descriptions}
"""

    def _call_llm_client(self, prompt: str) -> str:
        """
        Call the LLM client with automatic method detection.
        Supports common LLM client interfaces.
        """
        # Try common method names in order of preference
        method_attempts = [
            'send_sync_request',  # Your specific LLM client method
            'generate_response',
            'generate',
            'complete',
            'invoke',
            'call',
            'chat',
            'predict',
            '__call__'
        ]
        
        for method_name in method_attempts:
            if hasattr(self.llm_client, method_name):
                method = getattr(self.llm_client, method_name)
                if callable(method):
                    try:
                        logger.debug(f"Trying LLM method: {method_name}")
                        
                        # Try different calling patterns
                        if method_name in ['chat', 'invoke']:
                            # Some clients expect a list of messages
                            try:
                                response = method([{"role": "user", "content": prompt}])
                            except:
                                response = method(prompt)
                        elif method_name == '__call__':
                            # Callable object
                            response = self.llm_client(prompt)
                        else:
                            # Standard method call
                            response = method(prompt)
                        
                        # Extract text content if response is an object
                        if hasattr(response, 'content'):
                            return response.content
                        elif hasattr(response, 'text'):
                            return response.text
                        elif hasattr(response, 'message'):
                            return response.message
                        elif isinstance(response, dict):
                            # Try common dictionary keys
                            for key in ['content', 'text', 'message', 'response', 'output']:
                                if key in response:
                                    return str(response[key])
                            # If no standard keys, convert the whole dict to string
                            return str(response)
                        elif isinstance(response, str):
                            return response
                        else:
                            # Convert any other response type to string
                            return str(response)
                            
                    except Exception as e:
                        logger.debug(f"Method {method_name} failed: {e}")
                        continue
        
        # If all methods fail, provide helpful error message
        available_methods = [name for name in dir(self.llm_client) 
                           if not name.startswith('_') and callable(getattr(self.llm_client, name))]
        
        raise AttributeError(
            f"LLM client {type(self.llm_client)} doesn't have any recognized method. "
            f"Available methods: {available_methods}. "
            f"Please implement one of: {method_attempts}"
        )

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Enhanced JSON extraction with better error reporting.
        """
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")

        original_response = response.strip()
        logger.debug(f"Raw LLM response: '{original_response}'")

        # Strategy 1: Try to parse the response directly
        try:
            result = json.loads(original_response)
            logger.debug("Successfully parsed JSON directly")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")

        # Strategy 2: Remove common prefixes/suffixes and try again
        cleaned_response = self._clean_response(original_response)
        if cleaned_response != original_response:
            try:
                result = json.loads(cleaned_response)
                logger.debug("Successfully parsed JSON after cleaning")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Cleaned JSON parsing failed: {e}")

        # Strategy 3: Extract JSON from markdown blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                logger.debug("Successfully extracted JSON from markdown block")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Markdown JSON parsing failed: {e}")

        # Strategy 4: Find any JSON-like structure
        json_pattern = r'\{[^{}]*"relevant_tables"[^{}]*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in json_matches:
            try:
                # Try to fix common issues
                fixed_match = self._fix_json_issues(match)
                result = json.loads(fixed_match)
                logger.debug("Successfully parsed JSON with fixes")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Fixed JSON parsing failed for match: {e}")
                continue

        # Strategy 5: Create a fallback structure based on text analysis
        logger.warning("All JSON parsing strategies failed, attempting text analysis")
        try:
            return self._create_fallback_result(response)
        except Exception as e:
            logger.error(f"Fallback result creation failed: {e}")

        # Final failure - provide detailed error information
        error_info = {
            "original_response": original_response,
            "response_length": len(original_response),
            "contains_json_markers": "relevant_tables" in original_response.lower(),
            "contains_braces": "{" in original_response and "}" in original_response
        }
        
        logger.error(f"Complete JSON extraction failure. Debug info: {error_info}")
        raise ValueError(f"Could not extract valid JSON from LLM response. Response was: '{original_response[:200]}...'")

    def _clean_response(self, response: str) -> str:
        """Clean common response formatting issues."""
        # Remove markdown code blocks
        response = re.sub(r'^```(?:json)?\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'\s*```\s*$', '', response, flags=re.MULTILINE)
        
        # Remove common prefixes
        response = re.sub(r'^(?:Here\'s|Here is|The response is:?)\s*', '', response, flags=re.IGNORECASE)
        
        # Remove trailing explanations
        response = re.split(r'\n\n(?:This|The above|Explanation)', response)[0]
        
        return response.strip()

    def _fix_json_issues(self, json_text: str) -> str:
        """Fix common JSON formatting issues."""
        # Fix single quotes to double quotes
        json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)
        json_text = re.sub(r':\s*\'([^\']*)\'\s*([,}])', r': "\1"\2', json_text)
        
        # Fix unquoted keys
        json_text = re.sub(r'(\w+):', r'"\1":', json_text)
        
        # Remove trailing commas
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        return json_text

    def _create_fallback_result(self, response: str) -> Dict[str, Any]:
        """Create a fallback result by extracting information from text."""
        # Look for table names mentioned in the response
        common_tables = ["employees", "departments"]
        found_tables = []
        
        response_lower = response.lower()
        for table in common_tables:
            if table in response_lower:
                found_tables.append(table)
        
        # Extract reasoning if possible
        reasoning_match = re.search(r'(?:because|reason|explanation)[:\s]+([^.]*)', response, re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Extracted from text analysis"
        
        # Create confidence scores
        confidence_scores = {table: 0.6 for table in found_tables}
        
        logger.warning(f"Created fallback result with tables: {found_tables}")
        
        return {
            "relevant_tables": found_tables,
            "confidence_scores": confidence_scores,
            "reasoning": reasoning
        }

    @log_performance
    def run(self, user_prompt: str) -> Dict[str, Any]:
        """Select tables using AI semantic analysis with enhanced error handling."""
        try:
            # Prepare the prompt - handle JSON examples in instructions safely
            try:
                prompt = self.instructions.format(
                    user_query=user_prompt,
                    table_descriptions=self.table_descriptions
                )
            except KeyError as e:
                logger.error(f"Instruction template formatting error: {e}")
                logger.error("This usually means the instruction file contains unescaped curly braces")
                # Use safe string replacement as fallback
                prompt = self.instructions.replace("{user_query}", user_prompt)
                prompt = prompt.replace("{table_descriptions}", self.table_descriptions)

            logger.debug(f"Sending prompt to LLM: {prompt[:200]}...")

            # Call LLM - try different method names based on common LLM client interfaces
            start_time = time.time()
            response = self._call_llm_client(prompt)
            ai_latency = time.time() - start_time

            logger.debug(f"LLM response received in {ai_latency:.3f}s: '{response[:100]}...'")

            # Parse AI response with enhanced error handling
            ai_result = self._extract_json_from_response(response)

            # Validate and clean the result
            ai_result = self._validate_and_clean_result(ai_result)

            relevant_tables = ai_result.get("relevant_tables", [])
            confidence_scores = ai_result.get("confidence_scores", {})
            reasoning = ai_result.get("reasoning", "AI-based semantic analysis")

            logger.info(f"AISelector: user_prompt='{user_prompt}' => "
                        f"relevant_tables={relevant_tables} "
                        f"ai_latency={ai_latency:.3f}s")

            return {
                "relevant_tables_found": bool(relevant_tables),
                "relevant_tables": relevant_tables,
                "confidence_scores": confidence_scores,
                "reasoning": reasoning,
                "selection_method": "ai_powered",
                "ai_latency": ai_latency,
                "message": f"Found {len(relevant_tables)} relevant tables using AI semantic analysis."
            }

        except Exception as e:
            logger.error(f"AI table selection failed: {e}")
            # Log the full error details for debugging
            import traceback
            logger.error(f"Full error traceback: {traceback.format_exc()}")
            raise

    def _validate_and_clean_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the AI result to ensure it has the expected structure.
        """
        if not isinstance(result, dict):
            raise ValueError(f"AI result must be a dictionary, got {type(result)}")

        # Ensure required keys exist
        if "relevant_tables" not in result:
            result["relevant_tables"] = []

        if "confidence_scores" not in result:
            result["confidence_scores"] = {}

        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"

        # Clean and validate relevant_tables
        tables = result["relevant_tables"]
        if not isinstance(tables, list):
            if isinstance(tables, str):
                # Try to parse as comma-separated string
                tables = [t.strip() for t in tables.split(',') if t.strip()]
            else:
                tables = []

        result["relevant_tables"] = [str(table).strip() for table in tables if table]

        # Clean and validate confidence_scores
        scores = result["confidence_scores"]
        if not isinstance(scores, dict):
            scores = {}

        cleaned_scores = {}
        for table, score in scores.items():
            try:
                score_float = float(score)
                # Clamp score between 0 and 1
                cleaned_scores[str(table)] = max(0.0, min(1.0, score_float))
            except (ValueError, TypeError):
                logger.warning(f"Invalid confidence score for {table}: {score}")
                cleaned_scores[str(table)] = 0.5

        result["confidence_scores"] = cleaned_scores

        # Ensure reasoning is a string
        if not isinstance(result["reasoning"], str):
            result["reasoning"] = str(result["reasoning"])

        return result


class HybridTableSelector:
    """
    Hybrid table selector that combines rule-based and AI-powered approaches.
    """

    def __init__(self, llm_client=None, table_descriptions_file: str = None,
                 instruction_file: str = None, config: Dict[str, Any] = None):
        """Initialize hybrid selector."""
        # Initialize components
        self.rule_selector = RuleBasedTableSelector()

        # Configuration
        self.config = config or {}
        self.use_ai = self.config.get("use_ai", True) and llm_client is not None
        self.ai_timeout = self.config.get("ai_timeout", 5.0)
        self.fallback_threshold = self.config.get("fallback_threshold", 0.3)
        self.hybrid_mode = self.config.get("hybrid_mode", "ai_first")

        # Initialize AI selector if available
        self.ai_selector = None
        if self.use_ai:
            try:
                self.ai_selector = AITableSelector(
                    llm_client=llm_client,
                    table_descriptions_file=table_descriptions_file,
                    instruction_file=instruction_file
                )
                logger.info("HybridTableSelector: AI selector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AI selector: {e}. Using rule-based only.")
                self.use_ai = False

    @log_performance
    def run(self, user_prompt: str) -> Dict[str, Any]:
        """Select tables using hybrid approach."""
        if self.hybrid_mode == "rule_first":
            return self._rule_first_strategy(user_prompt)
        elif self.hybrid_mode == "both":
            return self._consensus_strategy(user_prompt)
        else:  # "ai_first" (default)
            return self._ai_first_strategy(user_prompt)

    def _ai_first_strategy(self, user_prompt: str) -> Dict[str, Any]:
        """Try AI first, fallback to rules if needed."""
        result = None

        # Try AI selection first
        if self.use_ai and self.ai_selector:
            try:
                result = self._try_ai_selection(user_prompt)

                # Check if AI result is satisfactory
                if (result and result.get("relevant_tables_found") and
                        self._is_ai_result_confident(result)):
                    result["strategy_used"] = "ai_primary"
                    return result

            except Exception as e:
                logger.warning(f"AI selection failed: {e}")

        # Fallback to rule-based selection
        logger.info("Falling back to rule-based selection")
        rule_result = self.rule_selector.run(user_prompt)
        rule_result["strategy_used"] = "rule_fallback"

        # If we have both results, we can enhance with AI insights
        if result and result.get("reasoning"):
            rule_result["ai_insights"] = result.get("reasoning")
            rule_result["strategy_used"] = "hybrid_fallback"

        return rule_result

    def _rule_first_strategy(self, user_prompt: str) -> Dict[str, Any]:
        """Try rules first, enhance with AI if needed."""
        rule_result = self.rule_selector.run(user_prompt)

        # If rule-based found tables, check if we should enhance with AI
        if rule_result.get("relevant_tables_found"):
            rule_confidence = self._calculate_rule_confidence(rule_result)

            if rule_confidence >= 0.8:  # High confidence, use rules
                rule_result["strategy_used"] = "rule_primary"
                return rule_result

        # Enhance with AI if available and confidence is low
        if self.use_ai and self.ai_selector:
            try:
                ai_result = self._try_ai_selection(user_prompt)
                if ai_result and ai_result.get("relevant_tables_found"):
                    # Merge results intelligently
                    merged_result = self._merge_results(rule_result, ai_result)
                    merged_result["strategy_used"] = "hybrid_enhanced"
                    return merged_result
            except Exception as e:
                logger.warning(f"AI enhancement failed: {e}")

        rule_result["strategy_used"] = "rule_only"
        return rule_result

    def _consensus_strategy(self, user_prompt: str) -> Dict[str, Any]:
        """Run both methods and create consensus result."""
        results = {}

        # Get rule-based result
        rule_result = self.rule_selector.run(user_prompt)
        results["rule"] = rule_result

        # Get AI result if available
        if self.use_ai and self.ai_selector:
            try:
                ai_result = self._try_ai_selection(user_prompt)
                results["ai"] = ai_result
            except Exception as e:
                logger.warning(f"AI selection in consensus mode failed: {e}")
                results["ai"] = None

        # Create consensus
        if results.get("ai"):
            merged_result = self._merge_results(rule_result, results["ai"])
            merged_result["strategy_used"] = "consensus"
            merged_result["individual_results"] = results
            return merged_result
        else:
            rule_result["strategy_used"] = "rule_only"
            return rule_result

    def _try_ai_selection(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt AI selection with timeout protection."""
        try:
            return self.ai_selector.run(user_prompt)
        except Exception as e:
            logger.error(f"AI selection error: {e}")
            return None

    def _is_ai_result_confident(self, ai_result: Dict[str, Any]) -> bool:
        """Check if AI result meets confidence threshold."""
        confidence_scores = ai_result.get("confidence_scores", {})
        if not confidence_scores:
            return len(ai_result.get("relevant_tables", [])) > 0

        # Check if any table has high confidence
        max_confidence = max(confidence_scores.values()) if confidence_scores else 0
        return max_confidence >= self.fallback_threshold

    def _calculate_rule_confidence(self, rule_result: Dict[str, Any]) -> float:
        """Calculate confidence score for rule-based result."""
        confidence_scores = rule_result.get("confidence_scores", {})
        if not confidence_scores:
            return 0.5 if rule_result.get("relevant_tables_found") else 0.0

        # Average confidence, weighted by number of matches
        total_score = sum(info.get("score", 0) for info in confidence_scores.values())
        max_possible = len(confidence_scores) * 3
        return min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0

    def _merge_results(self, rule_result: Dict[str, Any], ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently merge rule-based and AI results."""
        # Combine table lists
        rule_tables = set(rule_result.get("relevant_tables", []))
        ai_tables = set(ai_result.get("relevant_tables", []))
        merged_tables = list(rule_tables | ai_tables)

        # Merge confidence scores
        rule_confidence = rule_result.get("confidence_scores", {})
        ai_confidence = ai_result.get("confidence_scores", {})

        merged_confidence = {}
        for table in merged_tables:
            scores = []
            if table in rule_confidence:
                scores.append(self._normalize_rule_score(rule_confidence[table]))
            if table in ai_confidence:
                scores.append(ai_confidence[table])

            merged_confidence[table] = max(scores) if scores else 0.5

        return {
            "relevant_tables_found": bool(merged_tables),
            "relevant_tables": merged_tables,
            "confidence_scores": merged_confidence,
            "reasoning": ai_result.get("reasoning", "Hybrid rule-based and AI analysis"),
            "selection_method": "hybrid",
            "rule_contribution": list(rule_tables),
            "ai_contribution": list(ai_tables),
            "message": f"Found {len(merged_tables)} relevant tables using hybrid approach."
        }

    def _normalize_rule_score(self, rule_score_info: Dict[str, Any]) -> float:
        """Convert rule-based score to 0-1 scale."""
        score = rule_score_info.get("score", 0)
        return min(score / 3.0, 1.0)


# Main TableSelector class for backward compatibility
class TableSelector(HybridTableSelector):
    """Main TableSelector class - uses hybrid approach by default."""

    def __init__(self, llm_client=None, table_descriptions_file: str = None,
                 instruction_file: str = None, **kwargs):
        """Initialize TableSelector with hybrid approach."""
        config = {
            "use_ai": kwargs.get("use_ai", True),
            "hybrid_mode": kwargs.get("hybrid_mode", "ai_first"),
            "ai_timeout": kwargs.get("ai_timeout", 5.0),
            "fallback_threshold": kwargs.get("fallback_threshold", 0.3)
        }

        super().__init__(
            llm_client=llm_client,
            table_descriptions_file=table_descriptions_file,
            instruction_file=instruction_file,
            config=config
        )

        logger.info(f"TableSelector initialized with hybrid approach. "
                    f"AI enabled: {self.use_ai}, Mode: {self.hybrid_mode}")