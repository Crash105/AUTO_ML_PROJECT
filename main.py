import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import io
from openai import OpenAI
import os
from daytona import Daytona, DaytonaConfig

load_dotenv()


st.title("AutoML Agent")
st.markdown("Uplaod a csv file to get started")

uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

def summarize_dataset(dataframe: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataset for LLM context.
    
    This function creates a detailed text summary that includes:
    - Column data types and schema information
    - Missing value counts and data completeness
    - Cardinality (unique value counts) for each column
    - Statistical summaries for numeric columns
    - Sample data in CSV format
    
    Args:
        dataframe: The pandas dataframe to summarize
    Returns:
        A formatted string containing the dataset summary
    """
    try:
        buffer = io.StringIO()
        
        sample_rows = min(30, len(dataframe))
        
        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_csv = buffer.getvalue()
        
        dtypes = dataframe.dtypes.astype(str).to_dict()
        
        non_null_counts = dataframe.notnull().sum().to_dict()
        
        null_counts = dataframe.isnull().sum().to_dict()
        
        nunique = dataframe.nunique(dropna=True).to_dict()
        
        numeric_cols = [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
        
        desc = dataframe[numeric_cols].describe().to_dict() if numeric_cols else {
            
        }
        
        lines = []
        
        lines.append("Schema (dtype):")        
        for k, v in dtypes.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        
       
        lines.append("Null/Non-Null counts: ")
        for c in dataframe.columns:
            lines.append(f"- {c}: nulls = {int(null_counts[c])}, non_nulls={int(non_null_counts[c])}")
        lines.append("")
        
      
        
        
        lines.append("Cardinality (nunique Values)")
        for k,v in nunique.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")
        
        
        if desc:
            lines.append("Number summary stats (describe):")
            for col, stats in desc.items():
                stat_line = ", ".join([
                    f"{s}:{round(float(val), 4)}" if pd.notnull(val) else f"{s}:nan"
                    for s, val in stats.items()
                ])
                lines.append(f"- {col}: {stat_line}")
        lines.append("")  # optional blank line after all stats

        lines.append("Sample rows (CSV head): ")
        lines.append(sample_csv)    
        
        return "\n".join(lines)

    except Exception as e:
        
        return f"Error summarizing dataset: {e}"
        

def execute_in_daytona(script: str, csv_bytes: bytes):
    
    
    key = os.getenv("DAYTONA_API_KEY")
    if not key:
        raise ValueError("DAYTONA_API_KEY is not set")

    client = Daytona(DaytonaConfig(api_key=key))
    sandbox = client.create()
    exec_info = {}
    
    try:
        
        print("Hello1")
        sandbox.fs.upload_file(csv_bytes, "input.csv")
        print("Hello2")
        

       
        cmd = "python -u - <<'PY'\n" + script + "\nPY"
        
        result = sandbox.process.exec(cmd, timeout=600, env={"PYTHONUNBUFFERED": "1"})
        exec_info["exit_code"] = getattr(result, "exit_code", None)
        exec_info["stdout"] = getattr(result, "result", "")
        exec_info["stderr"] = getattr(result, "stderr", "")

        try:
            cleaned_bytes = sandbox.fs.download_file("cleaned.csv")
            return cleaned_bytes, exec_info
        except Exception as e:
            print(f"Error executing in daytona: {e}")
            return b"", exec_info

        
        
    except Exception as e:
        print(f"Error running Daytona script: {e}")
        exec_info['error'] = str(e)
        return b"", exec_info 
        
    
    


def built_cleaning_pipeline(df):
    data_summary = summarize_dataset(df)
    prompt = f"""
    You are an expert data scientist, specifically in field of data cleaning.
    You are given a dataframe and you need to clean the data. 
    The data summary is as follows:
    {data_summary}
    
    Please clean the data and return the clean data.
    Handle the following steps:
    - Missing values
    - Duplicate values
    - Outliers
    - Standardize the data accordingly
    - Use one-hot-encoding for categorical variables    

    Write a Python script to clean the data, based on the data summary provided, strictly in a json property called "script".

    ## IMPORTANT
    - Make sure to load in the data from the csv file called "input.csv"
    - The script should be a Python script that can be executed to clean the data.  
    - Make sure to save the cleaned data to a new csv file called "cleaned.csv".
    """
    return prompt


    

def get_openai_script(prompt: str) -> str:
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            
            messages = [
                {"role": "system", "content": (
                    "You are a senior data scientist. Always return a strict JSON object matching the users requested schema."
                )},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
            
        if not resp or not getattr(resp, "choices", None):
            return None
        text = resp.choices[0].message.content or ""
        
        try:
            data = json.loads(text)
            script_val = data.get("script")
            if isinstance(script_val, str) and script_val.strip():
                return script_val.strip()
        
        except Exception as e:
            print(f"Error getting script: {e}")
            pass
    except Exception as e:
        print(f"Error getting script: {e}")
        return None
    
    
            
         
        



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    selected_column = st.selectbox("Select a column to predict", df.columns.tolist(),
    help = "The column to predict")            
    
    button = st.button("Run AutoML") 
    
    if button:
        with st.spinner("Running AutoML..."):
            cleaning_prompt = built_cleaning_pipeline(df)
            with st.expander("Cleaning Prompt..."):
                st.write(cleaning_prompt)
            script = get_openai_script(cleaning_prompt)
            with st.expander("Script"):
                st.code(script)
            with st.status("Executing in Daytona..."):
                input_csv_bytes = df.to_csv(index=False).encode("utf-8")
                
                cleaned_bytes, exec_info = execute_in_daytona(script, input_csv_bytes)
                
                st.write("cleaned_bytes:", cleaned_bytes)

                
                st.write("exec_info:", exec_info)



                with st.expander("Execution Info"):
                    st.write(exec_info)
                with st.expander("Cleaned Data"):
                    cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                    st.dataframe(cleaned_df)
    
            
                    
            
            





