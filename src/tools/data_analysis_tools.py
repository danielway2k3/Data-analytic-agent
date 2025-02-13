from typing import Any, Dict, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt

class UniqueCountInput(BaseModel):
    """Input for UniqueCountTool"""
    column_name: str = Field(..., description="Name of the column to count unique values")

class MaxValueInput(BaseModel):
    """Input for MaxValueTool"""
    column_name: str = Field(..., description="Name of the column to find the max value")


class UniqueCountTool(BaseTool):
    name: str = "unique_count"
    description: str = """
    Counts unique values in a specified column of the dataframe.
    Input should be the column name as a string.
    Returns the count of unique values in that column.
    """
    args_schema: Type[BaseModel] = UniqueCountInput

    def _run(self, column_name: str) -> Dict[str, Any]:
        try:
            unique_count = df[column_name].nunique()
            return {"unique_count": unique_count}
        except Exception as e:
            return {"error": f"Error counting unique values: {str(e)}"}

    def _arun(self, column_name: str):
        raise NotImplementedError("This tool does not support async")

class MaxValueTool(BaseTool):
    name: str = "max_value"
    description: str = """
    Finds the maximum value in a specified column of the dataframe.
    Input should be the column name as a string.
    Returns the maximum value in that column.
    """
    args_schema: Type[BaseModel] = MaxValueInput

    def _run(self, column_name: str) -> Dict[str, Any]:
        try:
            max_value = df[column_name].max()
            return {"max_value": max_value}
        except Exception as e:
            return {"error": f"Error finding maximum value: {str(e)}"}

    def _arun(self, column_name: str):
        raise NotImplementedError("This tool does not support async")

def get_data_analysis_tools(df: pd.DataFrame) -> list:
    """Returns a list of all available data analysis tools."""
    return [
        UniqueCountTool(),
        MaxValueTool(),
    ]