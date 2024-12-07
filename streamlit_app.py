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

# Title and Description
st.title("💬 Financial Complaint Classifier")
st.write("A chatbot to classify customer complaints and create Jira tasks if needed.")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "classification_started" not in st.session_state:
    st.session_state.classification_started = False
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True
    )
if "product_described" not in st.session_state:
    st.session_state.product_described = False
if "problem_described" not in st.session_state:
    st.session_state.problem_described = False

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
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Display Chat History
if st.session_state.memory.chat_memory.messages:
    for message in st.session_state.memory.chat_memory.messages:
        st.chat_message(message["role"]).write(message["content"])

# Chat Input and Workflow
if user_input := st.chat_input("Describe your issue:"):
    st.session_state.memory.chat_memory.add_message({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Prompt to analyze user input
    product_categories = df1['Product'].unique()
    prompt = (
        f"You are a helpful assistant gathering information about financial complaints. Based on the following input:\n\n"
        f"'{user_input}'\n\n"
        f"Check if the user has specified:\n"
        f"1. A product (e.g., {', '.join(product_categories)}).\n"
        f"2. A problem or issue (e.g., 'stolen card', 'fraudulent transactions').\n\n"
        f"Respond naturally, asking only for missing information. If both details are present, acknowledge and prepare to proceed."
    )
    response = chat.predict(prompt).strip()
    st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # Update session state based on response
    if "product" in response.lower():
        st.session_state.product_described = True
    if "issue" in response.lower():
        st.session_state.problem_described = True

# Classification Process Trigger
if st.session_state.product_described and st.session_state.problem_described:
    if not st.session_state.classification_started:
        if st.button("Start Classification"):
            st.session_state.classification_started = True

            # Classification Process
            try:
                # Step 1: Classify by Product
                response_product = chat.predict(
                    f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
                    f"Complaint: {user_input}. Respond with the exact product as written there."
                )
                assigned_product = response_product.strip()

                # Step 2: Classify by Sub-product
                subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
                response_subproduct = chat.predict(
                    f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
                    f"Complaint: {user_input}. Respond with the exact sub-product as written there."
                )
                assigned_subproduct = response_subproduct.strip()

                # Step 3: Classify by Issue
                issue_options = df1[(df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)]['Issue'].unique()
                response_issue = chat.predict(
                    f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
                    f"Complaint: {user_input}. Respond with the exact issue as written there."
                )
                assigned_issue = response_issue.strip()

                # Display Classification Results
                classification_summary = (
                    f"Classification Results:\n"
                    f"- **Product**: {assigned_product}\n"
                    f"- **Sub-product**: {assigned_subproduct}\n"
                    f"- **Issue**: {assigned_issue}"
                )
                st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": classification_summary})
                st.chat_message("assistant").write(classification_summary)

            except Exception as e:
                error_message = f"Error during classification: {e}"
                st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": error_message})
                st.chat_message("assistant").write(error_message)
