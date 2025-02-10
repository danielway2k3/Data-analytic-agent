import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.logger.base_logger import BaseLogger
from src.models.llms import load_llm
from src.utils import excecute_plt_code

# load environment variables
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gemini-1.5-flash"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def process_query(agent, user_query):
    response = agent(user_query)
    
    action = response["intermediate_steps"][-1][0].tool_input["query"]
    
    if "plt" in action:
        st. write(response["output"])
        
        fig = excecute_plt_code(action, df=st.session_state.df)
        if fig:
            st.pyplot(fig)

        st.write("**Executed code:**")
        st.code(action)
        
        to_display_string = response["output"] + "\n" + f"```python\n{action}\n```"
        st.session_state.chat_history.append((user_query, to_display_string))

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
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file is not None:
        try:
            # Read csv file
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write("Data Preview:", st.session_state.df.head())

            # Create data analytics agent to query data
            agent = create_pandas_dataframe_agent(
                llm=llm, 
                df = st.session_state.df,
                agent_type="tool-calling",
                verbose=True, 
                allow_dangerous_code=True,
                return_intermediate_steps=True,
            )
            logger.info("### Successfully created data analytics agent ###")

            # Input query and process query
            user_query = st.text_input("Ask me anything about the data:")
            
            if st.button("Run query"):
                with st.spinner("Analyzing..."):
                    process_query(agent, user_query)
            

            # Display chat history
            st.subheader("Chat History")
            for role, message in st.session_state.chat_history:
                st.write(f"{role}: {message}")

        except Exception as e:
            logger.info(f"Error processing file: {str(e)}")  # Changed to info
            st.error("Error processing the file. Please make sure it's a valid CSV.")
            
    

if __name__ == "__main__":
    main()