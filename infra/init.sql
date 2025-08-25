-- Create extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for financial data
CREATE SCHEMA IF NOT EXISTS finance;

-- Create companies table
CREATE TABLE finance.companies (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create financial statements table
CREATE TABLE finance.financial_statements (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES finance.companies(id),
    statement_type VARCHAR(50) NOT NULL, -- income_statement, balance_sheet, cash_flow
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10), -- Q1, Q2, Q3, Q4, Annual
    data JSONB NOT NULL, -- Store financial data as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, statement_type, fiscal_year, fiscal_period)
);

-- Create stock prices table
CREATE TABLE finance.stock_prices (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES finance.companies(id),
    date DATE NOT NULL,
    open_price NUMERIC,
    close_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    volume BIGINT,
    adjusted_close NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- Create documents table for storing parsed documents
CREATE TABLE finance.documents (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES finance.companies(id),
    document_type VARCHAR(50) NOT NULL, -- annual_report, 10k, 10q, esg_report
    filing_date DATE,
    title VARCHAR(500),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector embeddings table
CREATE TABLE finance.document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES finance.documents(id),
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding VECTOR(1536), -- Default to OpenAI embedding size
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_companies_symbol ON finance.companies(symbol);
CREATE INDEX idx_financial_statements_company ON finance.financial_statements(company_id);
CREATE INDEX idx_financial_statements_type_year ON finance.financial_statements(statement_type, fiscal_year);
CREATE INDEX idx_stock_prices_company ON finance.stock_prices(company_id);
CREATE INDEX idx_stock_prices_date ON finance.stock_prices(date);
CREATE INDEX idx_documents_company ON finance.documents(company_id);
CREATE INDEX idx_documents_type ON finance.documents(document_type);
CREATE INDEX idx_document_embeddings_document ON finance.document_embeddings(document_id);
CREATE INDEX idx_document_embeddings_embedding ON finance.document_embeddings USING ivfflat (embedding vector_cosine_ops);
