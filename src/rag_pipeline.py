import psycopg2
import os
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from embeddings import EmbeddingProvider
from src.query_classifier import QueryClassifier

# Load environment variables
load_dotenv()

class RAGPipeline:
    """RAG pipeline for answering financial queries using structured and unstructured data."""
    
    def __init__(self):
        self.db_connection = None
        self._connect_to_db()
        self.embedding_provider = EmbeddingProvider()
        self.query_classifier = QueryClassifier()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self._setup_sql_chain()
    
    def _connect_to_db(self):
        """Establish database connection."""
        # Try to get DB_HOST from environment, with fallback logic
        db_host = os.getenv("DB_HOST", "localhost")
        
        # If running in Docker, the default should be "db" service name
        # Check if we're likely in a Docker container
        if os.path.exists("/.dockerenv") or "DBT_HOST" in os.environ:
            db_host = os.getenv("DBT_HOST", "db")
        
        try:
            self.db_connection = psycopg2.connect(
                host=db_host,
                database=os.getenv("DB_NAME", "llm_finance"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres"),
                port=os.getenv("DB_PORT", "5432")
            )
            print(f"Successfully connected to database at {db_host}")
        except Exception as e:
            print(f"Error connecting to database at {db_host}: {e}")
            raise
    
    def _setup_sql_chain(self):
        """Set up the SQL chain for structured data queries."""
        try:
            # Try to get DB_HOST from environment, with fallback logic
            db_host = os.getenv("DB_HOST", "localhost")
            
            # If running in Docker, the default should be "db" service name
            # Check if we're likely in a Docker container
            if os.path.exists("/.dockerenv") or "DBT_HOST" in os.environ:
                db_host = os.getenv("DBT_HOST", "db")
            
            # Create SQL database connection
            db_uri = f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@{db_host}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'llm_finance')}"
            self.db = SQLDatabase.from_uri(db_uri, schema="finance")
            
            # Create SQL chain
            self.sql_chain = create_sql_query_chain(
                self.llm,
                self.db
            )
            print(f"Successfully connected to SQL database at {db_host}")
        except Exception as e:
            print(f"Error setting up SQL chain: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with answer and supporting evidence
        """
        # Classify the query
        classification = self.query_classifier.classify_query(query)
        print(f"Query classification: {classification}")
        
        # Process based on classification
        if classification == "STRUCTURED":
            return self._process_structured_query(query)
        elif classification == "UNSTRUCTURED":
            return self._process_unstructured_query(query)
        else:  # HYBRID
            return self._process_hybrid_query(query)
    
    def _process_structured_query(self, query: str) -> Dict[str, Any]:
        """
        Process a structured query using SQL.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with answer and SQL query used
        """
        try:
            # Generate SQL query through LangChain
            sql_query = self.sql_chain.invoke({"question": query})
            
            # Clean up the SQL query by removing any problematic characters
            if isinstance(sql_query, str):
                # Remove markdown code block markers if present
                sql_query = sql_query.replace('```sql', '').replace('```', '')
                # Remove trailing whitespace and newlines
                sql_query = sql_query.strip()
                # Remove any extra semicolons
                sql_query = sql_query.replace(';', '')
                # Fix any Unicode characters that might cause issues
                sql_query = sql_query.replace('≥', '>=').replace('≤', '<=')
            
            # Execute the SQL query to get the actual result
            try:
                # Use the SQLDatabase tool to execute the query
                sql_tool = QuerySQLDataBaseTool(db=self.db)
                sql_result = sql_tool.invoke({"query": sql_query})
                
                # Format the answer with the actual result in a more readable way
                formatted_answer = self._format_sql_result(sql_result)
                
                # Generate a natural language response using the LLM
                natural_language_prompt = PromptTemplate(
                    template="""
                    You are a financial analyst assistant. Convert the following financial data into a natural language response.
                    
                    User Question: {query}
                    Financial Data: {formatted_data}
                    
                    Provide a clear, conversational answer that directly addresses the user's question using the financial data.
                    Include specific numbers and company names when relevant.
                    Do not mention technical details like SQL queries or database operations.
                    """,
                    input_variables=["query", "formatted_data"]
                )
                
                natural_language_chain = LLMChain(
                    llm=self.llm,
                    prompt=natural_language_prompt
                )
                
                # Generate natural language response
                natural_response = natural_language_chain.invoke({
                    "query": query,
                    "formatted_data": formatted_answer
                })
                
                # Use the natural language response as the answer
                final_answer = natural_response["text"]
                
            except Exception as exec_error:
                # If execution fails, fall back to just showing the query
                print(f"Error executing SQL query: {exec_error}")
                final_answer = f"Generated SQL query: {sql_query}"
                sql_result = final_answer
            
            return {
                "answer": final_answer,
                "evidence": {
                    "type": "structured",
                    "sql_query": sql_query,
                    "sql_result": sql_result
                },
                "classification": "STRUCTURED"
            }
            
        except Exception as e:
            print(f"Error processing structured query: {e}")
            return {
                "answer": "I encountered an error processing your structured query. Please try rephrasing.",
                "evidence": {
                    "type": "error",
                    "error": str(e)
                },
                "classification": "STRUCTURED"
            }
    
    def _process_unstructured_query(self, query: str) -> Dict[str, Any]:
        """
        Process an unstructured query using document search.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with answer and relevant document chunks
        """
        try:
            # Search for similar document chunks
            relevant_chunks = self._search_similar_chunks(query, top_k=3)
            
            # Create prompt with context
            context = "\n\n".join([chunk for chunk, _ in relevant_chunks])
            
            prompt_template = """
            You are a financial analyst assistant. Answer the question based on the provided context from company documents.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "query"]
            )
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )
            
            # Generate answer
            result = chain.invoke({
                "context": context,
                "query": query
            })
            
            return {
                "answer": result["text"],
                "evidence": {
                    "type": "unstructured",
                    "relevant_chunks": relevant_chunks
                },
                "classification": "UNSTRUCTURED"
            }
            
        except Exception as e:
            print(f"Error processing unstructured query: {e}")
            return {
                "answer": "I encountered an error processing your unstructured query. Please try rephrasing.",
                "evidence": {
                    "type": "error",
                    "error": str(e)
                },
                "classification": "UNSTRUCTURED"
            }
    
    def _process_hybrid_query(self, query: str) -> Dict[str, Any]:
        """
        Process a hybrid query using both structured and unstructured data.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with answer combining both data sources
        """
        try:
            # Get structured data
            structured_result = self._process_structured_query(query)
            
            # Get unstructured data
            unstructured_result = self._process_unstructured_query(query)
            
            # Combine information
            prompt_template = """
            You are a financial analyst assistant. Provide a comprehensive answer using both structured financial data and insights from company documents.
            
            Structured Data:
            {structured_answer}
            
            Document Insights:
            {unstructured_answer}
            
            Question: {query}
            
            Provide a comprehensive answer that combines both sources of information:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["structured_answer", "unstructured_answer", "query"]
            )
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )
            
            # Generate combined answer
            result = chain.invoke({
                "structured_answer": structured_result["answer"],
                "unstructured_answer": unstructured_result["answer"],
                "query": query
            })
            
            return {
                "answer": result["text"],
                "evidence": {
                    "type": "hybrid",
                    "structured": structured_result["evidence"],
                    "unstructured": unstructured_result["evidence"]
                },
                "classification": "HYBRID"
            }
            
        except Exception as e:
            print(f"Error processing hybrid query: {e}")
            return {
                "answer": "I encountered an error processing your hybrid query. Please try rephrasing.",
                "evidence": {
                    "type": "error",
                    "error": str(e)
                },
                "classification": "HYBRID"
            }
    
    def _search_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar document chunks using vector similarity.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk_text, similarity_score) tuples
        """
        cursor = self.db_connection.cursor()
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_provider.get_embedding(query)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # Search for similar chunks
            cursor.execute("""
                SELECT chunk_text, 1 - (embedding <=> %s::vector) as similarity
                FROM finance.document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, top_k))
            
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]
            
        except Exception as e:
            print(f"Error searching similar chunks: {e}")
            raise
        finally:
            cursor.close()
    
    def close_connection(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
    
    def _format_sql_result(self, sql_result):
        """
        Format SQL result for better readability.
        
        Args:
            sql_result: Raw SQL result from database
            
        Returns:
            str: Formatted result string
        """
        # Handle None results
        if sql_result is None:
            return "No data found."
        
        # Handle string results (which is what the SQL tool returns)
        if isinstance(sql_result, str):
            # Try to parse as Python literal if it looks like one
            try:
                import ast
                # Try to evaluate as Python literal
                parsed = ast.literal_eval(sql_result)
                return self._format_sql_result(parsed)
            except:
                # If it's not a Python literal, try to format it directly
                return self._format_string_result(sql_result)
        
        # Handle list/tuple results
        if isinstance(sql_result, (list, tuple)):
            # Convert any Decimal or other types to strings first
            converted_result = []
            for item in sql_result:
                if isinstance(item, (list, tuple)):
                    # Handle nested lists/tuples
                    converted_row = []
                    for sub_item in item:
                        # Handle Decimal types specifically
                        if hasattr(sub_item, 'is_decimal') or 'Decimal' in str(type(sub_item)):
                            converted_row.append(str(sub_item))
                        elif hasattr(sub_item, '__str__'):
                            converted_row.append(str(sub_item))
                        else:
                            converted_row.append(repr(sub_item))
                    converted_result.append(converted_row)
                else:
                    # Handle Decimal types specifically
                    if hasattr(item, 'is_decimal') or 'Decimal' in str(type(item)):
                        converted_result.append(str(item))
                    elif hasattr(item, '__str__'):
                        converted_result.append(str(item))
                    else:
                        converted_result.append(repr(item))
            
            return self._format_converted_result(converted_result)
        
        # Handle single values
        else:
            # Handle Decimal types specifically
            if hasattr(sql_result, 'is_decimal') or 'Decimal' in str(type(sql_result)):
                return f"Result: {str(sql_result)}"
            elif hasattr(sql_result, '__str__'):
                return f"Result: {str(sql_result)}"
            else:
                return f"Result: {repr(sql_result)}"
    
    def _format_string_result(self, sql_result_str):
        """
        Format string result from SQL tool.
        
        Args:
            sql_result_str: String representation of SQL result
            
        Returns:
            str: Formatted result string
        """
        # If it's a string that looks like a list of tuples, format it nicely
        if sql_result_str.startswith('[') and sql_result_str.endswith(']'):
            try:
                # Use regex to find and replace Decimal values with formatted numbers
                import re
                
                # Pattern to match Decimal('value') or Decimal("value")
                decimal_pattern = r"Decimal\('([^']+)'\)|Decimal\(\"([^\"]+)\"\)"
                
                def format_decimal(match):
                    # Get the decimal value (either from group 1 or group 2)
                    decimal_value = match.group(1) or match.group(2)
                    try:
                        value = float(decimal_value)
                        if value >= 1e12:
                            return f"${value/1e12:.2f} trillion"
                        elif value >= 1e9:
                            return f"${value/1e9:.2f} billion"
                        elif value >= 1e6:
                            return f"${value/1e6:.2f} million"
                        elif value >= 1e3:
                            return f"${value/1e3:.2f} thousand"
                        else:
                            return f"${value:.2f}"
                    except (ValueError, TypeError):
                        return decimal_value
                
                # Replace Decimal values with formatted numbers
                formatted_result = re.sub(decimal_pattern, format_decimal, sql_result_str)
                
                # Pattern to match datetime.date(year, month, day)
                date_pattern = r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)"
                
                def format_date(match):
                    year, month, day = match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                
                # Replace datetime.date with formatted dates
                formatted_result = re.sub(date_pattern, format_date, formatted_result)
                
                return f"Result: {formatted_result}"
            except Exception as e:
                # If formatting fails, return as is
                return f"Result: {sql_result_str}"
        else:
            # For simple string results, return as is
            return f"Result: {sql_result_str}"
    
    def _format_converted_result(self, converted_result):
        """
        Format converted result (all strings) for better readability.
        
        Args:
            converted_result: List of strings or list of lists of strings
            
        Returns:
            str: Formatted result string
        """
        if len(converted_result) == 0:
            return "No data found."
        
        # Handle list of lists (typical for SELECT queries)
        if all(isinstance(item, (list, tuple)) for item in converted_result):
            # Format as table
            formatted_rows = []
            for row in converted_result:
                formatted_row = []
                for item in row:
                    # Handle datetime objects
                    if 'datetime.date' in str(type(item)) or (hasattr(item, 'year') and hasattr(item, 'month') and hasattr(item, 'day')):
                        formatted_row.append(str(item))
                    # Handle Decimal types
                    elif 'Decimal' in str(type(item)):
                        try:
                            # Convert Decimal to float for formatting
                            value = float(str(item))
                            if value >= 1e12:
                                formatted_item = f"${value/1e12:.2f} trillion"
                            elif value >= 1e9:
                                formatted_item = f"${value/1e9:.2f} billion"
                            elif value >= 1e6:
                                formatted_item = f"${value/1e6:.2f} million"
                            elif value >= 1e3:
                                formatted_item = f"${value/1e3:.2f} thousand"
                            else:
                                formatted_item = f"${value:.2f}"
                            formatted_row.append(formatted_item)
                        except (ValueError, TypeError):
                            # If conversion fails, keep as is
                            formatted_row.append(str(item))
                    else:
                        # Try to format as number if it's numeric
                        try:
                            # Remove any quotes
                            clean_item = item.strip().strip("'\"")
                            # Try to convert to float
                            value = float(clean_item)
                            if value >= 1e12:
                                formatted_item = f"${value/1e12:.2f} trillion"
                            elif value >= 1e9:
                                formatted_item = f"${value/1e9:.2f} billion"
                            elif value >= 1e6:
                                formatted_item = f"${value/1e6:.2f} million"
                            elif value >= 1e3:
                                formatted_item = f"${value/1e3:.2f} thousand"
                            else:
                                formatted_item = f"${value:.2f}"
                            formatted_row.append(formatted_item)
                        except (ValueError, TypeError):
                            # Not a number, keep as is
                            formatted_row.append(item)
                formatted_rows.append(" | ".join(formatted_row))
            
            if len(formatted_rows) > 10:
                # If too many rows, just show first 10 and a summary
                return f"Showing first 10 rows out of {len(formatted_rows)} total:\n" + \
                       "\n".join(formatted_rows[:10]) + \
                       f"\n... and {len(formatted_rows) - 10} more rows"
            else:
                return "Results:\n" + "\n".join(formatted_rows)
        
        # Handle simple list
        else:
            formatted_items = []
            for item in converted_result:
                # Handle datetime objects
                if 'datetime.date' in str(type(item)) or (hasattr(item, 'year') and hasattr(item, 'month') and hasattr(item, 'day')):
                    formatted_items.append(str(item))
                # Handle Decimal types
                elif 'Decimal' in str(type(item)):
                    try:
                        # Convert Decimal to float for formatting
                        value = float(str(item))
                        if value >= 1e12:
                            formatted_item = f"${value/1e12:.2f} trillion"
                        elif value >= 1e9:
                            formatted_item = f"${value/1e9:.2f} billion"
                        elif value >= 1e6:
                            formatted_item = f"${value/1e6:.2f} million"
                        elif value >= 1e3:
                            formatted_item = f"${value/1e3:.2f} thousand"
                        else:
                            formatted_item = f"${value:.2f}"
                        formatted_items.append(formatted_item)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as is
                        formatted_items.append(str(item))
                else:
                    # Try to format as number if it's numeric
                    try:
                        # Remove any quotes
                        clean_item = item.strip().strip("'\"")
                        # Try to convert to float
                        value = float(clean_item)
                        if value >= 1e12:
                            formatted_item = f"${value/1e12:.2f} trillion"
                        elif value >= 1e9:
                            formatted_item = f"${value/1e9:.2f} billion"
                        elif value >= 1e6:
                            formatted_item = f"${value/1e6:.2f} million"
                        elif value >= 1e3:
                            formatted_item = f"${value/1e3:.2f} thousand"
                        else:
                            formatted_item = f"${value:.2f}"
                        formatted_items.append(formatted_item)
                    except (ValueError, TypeError):
                        # Not a number, keep as is
                        formatted_items.append(item)
            
            if len(formatted_items) > 10:
                # If too many items, just show first 10 and a summary
                return f"Showing first 10 items out of {len(formatted_items)} total:\n" + \
                       "\n".join(formatted_items[:10]) + \
                       f"\n... and {len(formatted_items) - 10} more items"
            else:
                return "Results: " + " | ".join(formatted_items)

# Example usage
if __name__ == "__main__":
    pipeline = RAGPipeline()
    
    try:
        # Test queries
        test_queries = [
            "What was Apple's revenue in 2023?",
            "What were Apple's main risks in 2023?",
            "Compare Tesla vs Microsoft revenue growth and their strategic risks"
        ]
        
        for query in test_queries:
            print(f"Processing query: {query}")
            result = pipeline.process_query(query)
            print(f"Answer: {result['answer']}")
            print(f"Classification: {result['classification']}")
            print("-" * 50)
        
    finally:
        pipeline.close_connection()
