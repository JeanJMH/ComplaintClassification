import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import pandas as pd
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain import hub


st.title("💬 Financial Complaint Classifier")
st.write("Classify customer complaints into Product, Sub-product, and Issue categories.")

# Load Dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Initialize OpenAI Chat
try:
    model_type = "gpt-4o-mini"
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model=model_type)
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Chat Input and Classification Workflow
if client_complaint := st.chat_input("Describe your issue:"):
    st.chat_message("user").write(client_complaint)

    # Step 1: Classify by Product
    product_categories = df1['Product'].unique()
    response_product = chat({"messages": [
        {"role": "system", "content": (
            f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
            "Respond with the exact product as written there."
        )},
        {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
    ]})
    assigned_product = response_product.strip()
    st.write(f"Assigned Product: {assigned_product}")

    # Step 2: Classify by Sub-product
    subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
    response_subproduct = chat({"messages": [
        {"role": "system", "content": (
            f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
            "Respond with the exact sub-product as written there."
        )},
        {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
    ]})
    assigned_subproduct = response_subproduct.strip()
    st.write(f"Assigned Sub-product: {assigned_subproduct}")

    # Step 3: Classify by Issue
    issue_options = df1[
        (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
    ]['Issue'].unique()
    response_issue = chat({"messages": [
        {"role": "system", "content": (
            f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
            "Respond with the exact issue as written there."
        )},
        {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
    ]})
    assigned_issue = response_issue.strip()
    st.write(f"Assigned Issue: {assigned_issue}")

    # Display Classification Results
    st.chat_message("assistant").write(
        f"Classification Results:\n- **Product**: {assigned_product}\n- **Sub-product**: {assigned_subproduct}\n- **Issue**: {assigned_issue}"
    )

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {assigned_product}")
    st.write(f"- **Sub-product**: {assigned_subproduct}")
    st.write(f"- **Issue**: {assigned_issue}")
