import os
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    st.error("❌ Missing required package: openai. Please install it with: pip install openai")
    st.stop()

try:
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    st.error("❌ Missing required package: sqlalchemy. Please install it with: pip install sqlalchemy")
    st.stop()

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("❌ Missing required package: plotly. Please install it with: pip install plotly")
    st.stop()

env_path = Path('/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env')
if not env_path.exists():
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

st.set_page_config(
    page_title="SQL Chat Agent",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .query-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'db_connection_string' not in st.session_state:
    st.session_state.db_connection_string = None
if 'db_type' not in st.session_state:
    st.session_state.db_type = None

def init_openai_client():
    """Initialize OpenAI client with DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return None
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )

def get_database_schema(engine) -> str:
    """Get database schema information"""
    inspector = inspect(engine)
    schema_info = []
    
    for table_name in inspector.get_table_names():
        schema_info.append(f"\nTable: {table_name}")
        columns = inspector.get_columns(table_name)
        for col in columns:
            schema_info.append(f"  - {col['name']} ({col['type']})")
    
    return "\n".join(schema_info)

def generate_sql_from_natural_language(natural_language: str, schema: str, db_type: str, client: OpenAI) -> str:
    """Convert natural language query to SQL using DeepSeek"""
    
    db_instructions = ""
    if db_type == "MySQL":
        db_instructions = """
IMPORTANT MySQL-specific syntax rules:
- Use DATE_FORMAT(date_column, '%Y-%m-01') instead of DATE_TRUNC() for month truncation
- Use DATE_FORMAT(date_column, '%Y-%m') for grouping by year-month
- Use YEAR(date_column), MONTH(date_column) for date extraction
- Use DATE_ADD() or DATE_SUB() for date arithmetic
- Use CONCAT() for string concatenation
- Use IFNULL() or COALESCE() for null handling
- Do NOT use PostgreSQL functions like DATE_TRUNC, EXTRACT with date_part, etc.
- Use LIMIT instead of FETCH FIRST
- Use backticks (`) for identifiers if they contain special characters
"""
    elif db_type == "PostgreSQL":
        db_instructions = """
IMPORTANT PostgreSQL-specific syntax rules:
- Use DATE_TRUNC('month', date_column) for month truncation
- Use EXTRACT() or DATE_PART() for date extraction
- Use || for string concatenation
- Use COALESCE() for null handling
- Use LIMIT or FETCH FIRST for limiting results
"""
    elif db_type == "SQLite":
        db_instructions = """
IMPORTANT SQLite-specific syntax rules:
- Use strftime('%Y-%m', date_column) for date formatting
- Use date() function for date operations
- Use || for string concatenation
- Use IFNULL() or COALESCE() for null handling
- Use LIMIT for limiting results
"""
    
    prompt = f"""You are an expert SQL query generator. Given a database schema and a natural language question, generate a valid SQL query.

Database Type: {db_type}
{db_instructions}

Database Schema:
{schema}

User Question: {natural_language}

Instructions:
1. Generate ONLY the SQL query, no explanations
2. Do not include markdown code blocks or backticks
3. Use proper SQL syntax for {db_type} database (follow the rules above)
4. Make sure the query is safe and only reads data (SELECT queries only)
5. Return only the SQL query, nothing else
6. Use database-specific functions as specified above

SQL Query:"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert SQL query generator. Always return only the SQL query without any markdown formatting or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = sql_query.strip()
        
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL: {str(e)}")
        return None

def is_safe_query(sql_query: str) -> bool:
    """Check if query is safe (only SELECT statements)"""
    sql_clean = re.sub(r'--.*', '', sql_query, flags=re.MULTILINE)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
    sql_clean = sql_clean.strip().upper()
    
    if not sql_clean.startswith('SELECT'):
        return False
    
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
    for keyword in dangerous_keywords:
        if keyword in sql_clean:
            return False
    
    return True

def execute_sql_query(engine, sql_query: str) -> Optional[pd.DataFrame]:
    """Execute SQL query and return results as DataFrame"""
    if not is_safe_query(sql_query):
        st.error("Only SELECT queries are allowed for security reasons.")
        return None
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
            return df
    except SQLAlchemyError as e:
        st.error(f"SQL Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def generate_natural_language_response(query: str, results: pd.DataFrame, client: OpenAI) -> str:
    """Convert SQL results to natural language explanation"""
    if results.empty:
        return "The query returned no results."
    
    results_summary = results.head(20).to_string()  
    
    prompt = f"""The user asked: "{query}"

The SQL query returned the following results:
{results_summary}

Please provide a clear, natural language explanation of these results. Be concise and highlight key insights. If there are many rows, mention the total count."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst that explains SQL query results in clear, natural language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Query executed successfully. Found {len(results)} rows."

def should_create_visualization(df: pd.DataFrame) -> bool:
    """Determine if results should be visualized"""
    if df.empty or len(df) == 0:
        return False
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) == 0:
        return False
    
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        return True
    if len(numeric_cols) >= 1 and (len(categorical_cols) >= 1 or len(date_cols) >= 1):
        return True
    if len(numeric_cols) == 1 and len(df) <= 100:
        return True
    
    return False

def create_visualization(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create appropriate visualization based on data"""
    if df.empty:
        return None
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    try:
        if len(date_cols) >= 1 and len(numeric_cols) >= 1:
            date_col = date_cols[0]
            numeric_col = numeric_cols[0]
            fig = px.line(df, x=date_col, y=numeric_col, 
                         title=f"{numeric_col} over Time")
            return fig
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if len(df[cat_col].unique()) <= 20:
                fig = px.bar(df, x=cat_col, y=num_col,
                           title=f"{num_col} by {cat_col}")
                fig.update_xaxes(tickangle=45)
                return fig
        
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
            return fig
        
        if len(numeric_cols) == 1:
            num_col = numeric_cols[0]
            if len(df) <= 50:
                fig = px.bar(df, y=num_col, title=f"Distribution of {num_col}")
            else:
                fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
            return fig
        
        if len(numeric_cols) > 2:
            fig = go.Figure()
            for col in numeric_cols[:5]:  
                fig.add_trace(go.Scatter(
                    y=df[col],
                    mode='lines+markers',
                    name=col
                ))
            fig.update_layout(title="Multiple Metrics Comparison",
                            xaxis_title="Index",
                            yaxis_title="Value")
            return fig
        
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
        return None
    
    return None

def connect_to_database(connection_string: str, db_type: str):
    """Connect to database using SQLAlchemy"""
    try:
        if db_type == "SQLite":
            engine = create_engine(connection_string, echo=False)
        elif db_type == "PostgreSQL":
            engine = create_engine(connection_string, echo=False)
        elif db_type == "MySQL":
            if hasattr(st.session_state, 'mysql_host') and st.session_state.mysql_host:
                host = st.session_state.mysql_host
                port = st.session_state.mysql_port
                database = st.session_state.mysql_database
                username = st.session_state.mysql_username
                password = st.session_state.mysql_password
                
                base_url = f"mysql+pymysql://{host}:{port}/{database}"
                engine = create_engine(
                    base_url,
                    echo=False,
                    connect_args={
                        "user": username,
                        "password": password,
                        "charset": "utf8mb4",
                        "connect_timeout": 10
                    }
                )
            else:
                engine = create_engine(
                    connection_string, 
                    echo=False,
                    connect_args={
                        "charset": "utf8mb4",
                        "connect_timeout": 10
                    }
                )
        else:
            st.error(f"Unsupported database type: {db_type}")
            return None
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        st.session_state.db_engine = engine
        st.session_state.db_connection_string = connection_string
        st.session_state.db_type = db_type
        return engine
    except SQLAlchemyError as e:
        error_msg = str(e)
        if "Access denied" in error_msg or "1045" in error_msg:
            st.error(f"❌ **Authentication Failed** - Access denied for user")
            st.warning("The password you entered is incorrect for this MySQL user.")
            st.info("**Troubleshooting Steps:**")
            st.markdown("""
            1. **Verify the password is correct** - Test with MySQL command line:
               ```bash
               /usr/local/mysql-9.4.0-macos15-arm64/bin/mysql -u root -p -h localhost
               ```
            2. **Reset MySQL root password** if you've forgotten it
            3. **Create a new MySQL user** with the password you want:
               ```sql
               CREATE USER 'analytics_user'@'localhost' IDENTIFIED BY 'Test@123';
               GRANT ALL PRIVILEGES ON analytics_chatdb.* TO 'analytics_user'@'localhost';
               FLUSH PRIVILEGES;
               ```
            4. Check the setup guide: `mysql_setup_guide.md` in this directory
            """)
        else:
            st.error(f"Error connecting to database: {error_msg}")
        return None
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

st.markdown('<h1 class="main-header">SQL Chat Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions in natural language and get SQL-powered insights with visualizations</p>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Database Configuration")
    
    if DEEPSEEK_API_KEY:
        st.success("✓ DeepSeek API key loaded")
    else:
        st.error("✗ DeepSeek API key not found in .env file")
    
    st.markdown("---")
    
    db_type = st.selectbox(
        "Database Type",
        ["MySQL", "PostgreSQL", "SQLite"],
        index=0
    )
    
    if db_type == "MySQL":
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="3306")
        database = st.text_input("Database", value="analytics_chatdb")
        username = st.text_input("Username", value="root")
        password = st.text_input("Password", type="password")
        if all([host, port, database, username, password]):
            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password)
            connection_string = f"mysql+pymysql://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
        else:
            connection_string = None
    elif db_type == "PostgreSQL":
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="5432")
        database = st.text_input("Database", value="analytics_chatdb")
        username = st.text_input("Username", value="postgres")
        password = st.text_input("Password", type="password")
        if all([host, port, database, username, password]):
            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password)
            connection_string = f"postgresql://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
        else:
            connection_string = None
    else:  
        db_path = st.text_input(
            "Database Path",
            value="analytics_chatdb.db",
            help="Path to your SQLite database file"
        )
        if db_path:
            connection_string = f"sqlite:///{db_path}"
    
    if st.button("Connect to Database"):
        if connection_string:
            if db_type == "MySQL":
                st.session_state.mysql_host = host
                st.session_state.mysql_port = port
                st.session_state.mysql_database = database
                st.session_state.mysql_username = username
                st.session_state.mysql_password = password
            engine = connect_to_database(connection_string, db_type)
            if engine:
                st.success("✓ Connected to database")
                try:
                    schema = get_database_schema(engine)
                    with st.expander("Database Schema"):
                        st.code(schema, language="text")
                except Exception as e:
                    st.warning(f"Could not load schema: {str(e)}")
        else:
            st.error("Please fill in all database connection details")
    
    if st.session_state.db_engine:
        st.success("✓ Database connected")
        if st.button("Disconnect"):
            st.session_state.db_engine = None
            st.session_state.db_connection_string = None
            st.session_state.db_type = None
            st.rerun()

if not DEEPSEEK_API_KEY:
    st.error("Please configure DEEPSEEK_API_KEY in your root .env file")
elif not st.session_state.db_engine:
    st.info("👈 Please connect to a database using the sidebar")
else:
    client = init_openai_client()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sql_query" in message:
                with st.expander("View SQL Query"):
                    st.code(message["sql_query"], language="sql")
            if "dataframe" in message and not message["dataframe"].empty:
                st.dataframe(message["dataframe"], use_container_width=True)
            if "visualization" in message and message["visualization"]:
                st.plotly_chart(message["visualization"], use_container_width=True)
    
    user_query = st.chat_input("Ask a question about your database...")
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                schema = get_database_schema(st.session_state.db_engine)
                
                db_type = st.session_state.db_type
                sql_query = generate_sql_from_natural_language(user_query, schema, db_type, client)
                
                if not sql_query:
                    st.error("Failed to generate SQL query")
                    st.stop()
                
                with st.expander("Generated SQL Query"):
                    st.code(sql_query, language="sql")
                
                with st.spinner("Executing query..."):
                    results_df = execute_sql_query(st.session_state.db_engine, sql_query)
                
                if results_df is None:
                    st.error("Failed to execute SQL query")
                    st.stop()
                
                if not results_df.empty:
                    st.dataframe(results_df, use_container_width=True)
                    
                    with st.spinner("Generating response..."):
                        natural_response = generate_natural_language_response(
                            user_query, results_df, client
                        )
                    st.markdown(natural_response)
                    
                    visualization = None
                    if should_create_visualization(results_df):
                        with st.spinner("Creating visualization..."):
                            visualization = create_visualization(results_df)
                        if visualization:
                            st.plotly_chart(visualization, use_container_width=True)
                    
                    message_data = {
                        "role": "assistant",
                        "content": natural_response,
                        "sql_query": sql_query,
                        "dataframe": results_df,
                        "visualization": visualization
                    }
                    st.session_state.messages.append(message_data)
                else:
                    response = "The query returned no results."
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sql_query": sql_query
                    })

