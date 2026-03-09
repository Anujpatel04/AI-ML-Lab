# SQL Chat Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

An intelligent SQL agent that converts natural language questions into database-specific SQL queries, executes them, and provides natural language responses with automatic visualizations. Powered by DeepSeek AI.

## Features

- Natural language to SQL conversion with database-specific syntax (MySQL, PostgreSQL, SQLite)
- Automatic query result visualization (bar charts, line charts, scatter plots, histograms)
- Natural language explanations of query results
- Schema-aware query generation
- Read-only access (SELECT queries only) for security
- Professional Streamlit interface

## Prerequisites

- Python 3.8+
- DeepSeek API key
- Database (SQLite, PostgreSQL, or MySQL)

## Installation

```bash
cd AI_AGENTS/ChatWith_SQL_Locally
pip install -r requirements.txt
```

## Configuration

Add your DeepSeek API key to the root `.env` file:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1  # Optional
```

The application automatically loads API keys from the root `.env` file.

## Usage

### Running the Application

```bash
streamlit run sql_agent.py
```

Access at `http://localhost:8501`

### Connecting to Database

1. Select database type (MySQL, PostgreSQL, or SQLite)
2. Enter connection details (host, port, database, username, password)
3. Click "Connect to Database"

### Asking Questions

Ask questions in natural language:

- "Show me all customers from New York"
- "What are the top 10 products by sales?"
- "Display monthly revenue trends"
- "What is the total sales for each product category?"

The agent will generate SQL, execute it, display results, explain them in natural language, and create visualizations when appropriate.

## How It Works

1. User asks a question in natural language
2. System retrieves database schema
3. DeepSeek AI generates database-specific SQL query
4. Query executes against the database
5. Results are displayed and explained in natural language
6. Visualizations are automatically created when suitable

## Database-Specific SQL

The agent generates syntax-aware SQL:
- **MySQL**: `DATE_FORMAT()`, `YEAR()`, `MONTH()` for dates
- **PostgreSQL**: `DATE_TRUNC()`, `EXTRACT()` for dates
- **SQLite**: `strftime()` for date formatting

## Dependencies

- streamlit, openai, sqlalchemy, pandas, plotly
- python-dotenv, pymysql, psycopg2-binary, cryptography

See `requirements.txt` for complete list with versions.

## Security

- Read-only queries (SELECT only)
- Credentials stored in session state (not persisted)
- API keys loaded from environment variables
- Special characters in passwords are URL-encoded

**SQL Generation**: Ensure correct database type is selected. Review generated SQL in the expandable section.

**API Key**: Verify `DEEPSEEK_API_KEY` is set in root `.env` file.

## License

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab) - MIT License
