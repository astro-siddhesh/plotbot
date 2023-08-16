from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from tempfile import NamedTemporaryFile
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import pandas as pd    
import matplotlib

matplotlib.use('TkAgg')
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(api_token = API_KEY)
pandas_ai = PandasAI(llm) 

st.title("Prompt-driven analysis with PandasAI")
uploaded_file = st.file_uploader("upload a csv file for analysis" , type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))
    
    prompt = st.text_area("enter your question:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("generating response ,  please wait....."):
                st.write(pandas_ai.run(df , prompt = prompt))
        else :
            st.warning("enter your question.")