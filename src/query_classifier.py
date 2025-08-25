from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Union
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QueryClassifier:
    """Classify user queries to determine the appropriate processing approach."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self._create_chain()
    
    def _create_chain(self):
        """Create the classification chain."""
        prompt_template = """
        You are an AI assistant that classifies user queries related to financial analysis of companies.
        Classify the following query into one of these categories:
        
        1. STRUCTURED: Queries that can be answered with financial data (revenue, profits, stock prices, etc.)
        2. UNSTRUCTURED: Queries that require analysis of documents (annual reports, 10-Ks, risk factors, etc.)
        3. HYBRID: Queries that require both structured financial data and document analysis
        
        Examples:
        - "What was Apple's revenue in 2023?" -> STRUCTURED
        - "What were Apple's main risks in 2023?" -> UNSTRUCTURED
        - "Compare Tesla vs Microsoft revenue growth and their strategic risks" -> HYBRID
        
        Query: {query}
        
        Respond with ONLY the category name (STRUCTURED, UNSTRUCTURED, or HYBRID) in uppercase.
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query"]
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
    
    def classify_query(self, query: str) -> str:
        """
        Classify a user query.
        
        Args:
            query (str): User query to classify
            
        Returns:
            str: Classification result (STRUCTURED, UNSTRUCTURED, or HYBRID)
        """
        try:
            result = self.chain.invoke({"query": query})
            classification = result["text"].strip().upper()
            
            # Validate classification
            if classification not in ["STRUCTURED", "UNSTRUCTURED", "HYBRID"]:
                # Default to HYBRID if classification is unclear
                return "HYBRID"
            
            return classification
            
        except Exception as e:
            print(f"Error classifying query: {e}")
            # Default to HYBRID if there's an error
            return "HYBRID"

# Example usage
if __name__ == "__main__":
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        "What was Apple's revenue in 2023?",
        "What were Apple's main risks in 2023?",
        "Compare Tesla vs Microsoft revenue growth and their strategic risks",
        "Show me the stock price trend for NVIDIA over the last year",
        "Summarize Amazon's ESG commitments"
    ]
    
    for query in test_queries:
        classification = classifier.classify_query(query)
        print(f"Query: {query}")
        print(f"Classification: {classification}")
        print("-" * 50)
