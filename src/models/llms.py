from langchain_google_genai import ChatGoogleGenerativeAI

def load_llm(model_name):
    """Load Large Language Model.

    Args:
        model_name (str): The name of the model to load.

    Raises:
        ValueError: If the model_name is not recognized.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI configured for the specified model.
    """
    if model_name == "gemini-1.5-flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,
            max_output_tokens=1000,
        )
        return llm
    
    elif model_name == "gemini-2.0-flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            max_output_tokens=1000,
        )
        return llm
    else:
        raise ValueError("Model name not found\
            Please choose from 'gemini-1.5-flash' or 'gemini-2.0-flash'")