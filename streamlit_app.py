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
st.write("Classify customer complaints into Product, Sub-product, and Issue categories and create Jira tasks if needed.")

# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# Helper function for evaluating input sufficiency
def evaluate_input_sufficiency(chat, input_text, product_categories):
    """
    Use a language model to evaluate if the input text has enough information for classification.
    """
    evaluation_prompt = (
        f"You are an intelligent assistant. Evaluate if the following customer complaint contains enough "
        f"information to classify the issue into a product and problem category. "
        f"The product categories are: {product_categories.tolist()}.\n\n"
        f"The complaint is: '{input_text}'.\n\n"
        f"Respond with 'Sufficient' if it contains enough information, or 'Insufficient' "
        f"if it lacks clarity or detail."
    )
    
    response = chat.predict_messages([
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": input_text}
    ])
    return response.content.strip()

# Helper function for classification
def classify_complaint(chat, prompt, input_text):
    """
    Classify complaints into categories (Product, Sub-product, or Issue) using the ChatOpenAI model.
    """
    response = chat.predict_messages([
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text}
    ])
    return response.content.strip()

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Chat Input and Workflow
if client_complaint := st.chat_input("Describe your issue:"):
    # Add user's message to chat history
    st.session_state.chat_history.append({"role": "user", "content": client_complaint})
    st.chat_message("user").write(client_complaint)

    # Evaluate input sufficiency
    product_categories = df1['Product'].unique()
    sufficiency_result = evaluate_input_sufficiency(chat, client_complaint, product_categories)

    if sufficiency_result.lower() == "sufficient":
        st.session_state.chat_history.append({"role": "assistant", "content": "Thank you! Starting the classification process..."})
        st.chat_message("assistant").write("Thank you! Starting the classification process...")

        # Step 1: Classify by Product
        product_prompt = (
            f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
            "Respond with the exact product as written there."
        )
        assigned_product = classify_complaint(chat, product_prompt, client_complaint)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Assigned Product: {assigned_product}"})
        st.write(f"Assigned Product: {assigned_product}")

        # Step 2: Classify by Sub-product
        subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
        subproduct_prompt = (
            f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
            "Respond with the exact sub-product as written there."
        )
        assigned_subproduct = classify_complaint(chat, subproduct_prompt, client_complaint)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Assigned Sub-product: {assigned_subproduct}"})
        st.write(f"Assigned Sub-product: {assigned_subproduct}")

        # Step 3: Classify by Issue
        issue_options = df1[
            (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
        ]['Issue'].unique()
        issue_prompt = (
            f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
            "Respond with the exact issue as written there."
        )
        assigned_issue = classify_complaint(chat, issue_prompt, client_complaint)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Assigned Issue: {assigned_issue}"})
        st.write(f"Assigned Issue: {assigned_issue}")

        # Display Classification Results
        summary = f"Classification Results:\n- **Product**: {assigned_product}\n- **Sub-product**: {assigned_subproduct}\n- **Issue**: {assigned_issue}"
        st.session_state.chat_history.append({"role": "assistant", "content": summary})
        st.chat_message("assistant").write(summary)

    else:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "It seems your input lacks sufficient details. Could you specify the product (e.g., 'Credit Card', 'Savings Account') and describe the issue more clearly?"}
        )
        st.chat_message("assistant").write(
            "It seems your input lacks sufficient details. Could you specify the product (e.g., 'Credit Card', 'Savings Account') and describe the issue more clearly?"
        )
