import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

def main():
    # Setup streamlit interface
    st.set_page_config(
        page_title="Interactive Visualization Tool",
        page_icon="📊",
        layout="wide",
    )

    st.header("Interactive Visualization Tool")
    st.write("### Welcome to the Interactive Visualization Tool!")
    
    # Render pywalker
    if st.session_state.get("df") is not None:
        pyg_app = StreamlitRenderer(st.session_state.df)
        pyg_app.explorer()
        
    else:
        st.info("Please upload a CSV file to get started.")
    
if __name__ == "__main__":
    main()