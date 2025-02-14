import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

from src.logger.base_logger import BaseLogger
from src.models.llms import load_llm
from src.utils import execute_plt_code
from src.tools.data_analysis_tools import get_data_analysis_tools

# load environment variables
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gemini-1.5-flash"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def process_query(da_agent, query):
    response = da_agent(query)

    # Extract code from intermediate steps
    code_to_execute = None
    if "intermediate_steps" in response:
        for step in response["intermediate_steps"]:
            if step[0].tool == "python_repl_ast":
                tool_input = step[0].tool_input
                if isinstance(tool_input, dict):
                    code_to_execute = tool_input.get("query", "")
                else:
                    code_to_execute = tool_input
                break

    if code_to_execute and ("plt" in code_to_execute or "plot" in code_to_execute):
        st.write(response["output"])
        
        cleaned_code = code_to_execute.strip('```python').strip('```').strip()
        
        fig = execute_plt_code(cleaned_code, df=st.session_state.df)
        if fig:
            st.pyplot(fig)
            plt.close() 
        
        st.write("**Executed Code:**")
        st.code(cleaned_code, language="python")
        
        to_display_string = response["output"] + "\n```python\n" + cleaned_code + "\n```"
        st.session_state.history.append((query, to_display_string))
        logger.info(f"### Successfully executed query: {query} ###")
    else:
        try:
            intermediate_result = response["intermediate_steps"][0][1]
            # Check if result is already a DataFrame
            if isinstance(intermediate_result, pd.DataFrame):
                st.write(intermediate_result)
            elif isinstance(intermediate_result, str):
                # Try to convert string to DataFrame if it looks like tabular data
                if "describe" or "info" in response["output"].lower():
                    # Convert describe() output string to DataFrame
                    try:
                        # Split the string into lines and parse as CSV
                        lines = intermediate_result.strip().split('\n')
                        import io
                        df_result = pd.read_csv(io.StringIO(intermediate_result), sep='\s+')
                        st.write(df_result)
                    except:
                        # If conversion fails, display as formatted text
                        st.text(intermediate_result)
                else:
                    st.write(intermediate_result)
            else:
                # Try to convert to DataFrame or display as is
                try:
                    st.write(pd.DataFrame(intermediate_result))
                except:
                    st.write(intermediate_result)
            
            st.write(response["output"])
            st.session_state.history.append((query, response["output"]))
            logger.info(f"### Successfully executed query: {query} ###")
        except Exception as e:
            logger.error(f"Error processing result: {str(e)}")
            st.error("Error processing the result. Displaying raw output instead.")
            st.write(response["output"])

def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**Query: {i+1}:** {q}")
        st.markdown(f"**Response: {i+1}:** {r}")
        st.markdown("---")


def main():
    # Setup streamlit interface
    st.set_page_config(
        page_title="Smart Data Analytics tool",
        page_icon="📈",
        layout="centered",
    )
    st.header("Smart Data Analytics tool")
    st.write("This tool helps you analyze your data using a large language model.")

    # Load llms model
    try:
        llm = load_llm(MODEL_NAME)
        logger.info(f"### Successfully loaded {MODEL_NAME} ###")
    except Exception as e:
        logger.info(f"Error loading model: {str(e)}")
        st.error("Failed to load the model. Please check your API key.")
        return

    # Upload csv file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    # Initial chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Read csv file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.write("Data Preview:", df.head())

            # Create data analytics agent to query data
            agent = create_pandas_dataframe_agent(
                llm=llm, 
                df = df,
                verbose=True, 
                allow_dangerous_code=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                prefix="""You are working with a pandas dataframe 'df' that contains the complete dataset.
                Always use the provided 'df' and DO NOT create new sample dataframes.
                The dataframe is already loaded and contains {num_rows} rows.""".format(num_rows=len(df)),
            )
            logger.info("### Successfully created data analytics agent ###")

            # Input query and process query
            user_query = st.text_input("Ask me anything about the data:")
            
            if st.button("Run query"):
                with st.spinner("Analyzing..."):
                    process_query(agent, user_query)
            

            # Display chat history
            st.divider()
            display_chat_history()

        except Exception as e:
            logger.info(f"Error processing file: {str(e)}")  # Changed to info
            st.error("Error processing the file. Please make sure it's a valid CSV.")
            
    

if __name__ == "__main__":
    main()