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
if "context_collected" not in st.session_state:
    st.session_state.context_collected = False
if "product" not in st.session_state:
    st.session_state.product = None
if "issue" not in st.session_state:
    st.session_state.issue = None

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

# Helper function to check missing details
def check_missing_details(chat, user_input, product_categories):
    prompt = (
        f"You are a helpful assistant gathering information to classify customer complaints. "
        f"Determine if the following input provides details about both a product and an issue:\n\n"
        f"Input: '{user_input}'\n\n"
        f"Products: {', '.join(product_categories)}\n"
        f"Issues: e.g., 'fraudulent transactions', 'stolen card', 'unauthorized charges'.\n\n"
        f"Respond with one of the following:\n"
        f"- 'Missing: Product'\n"
        f"- 'Missing: Issue'\n"
        f"- 'Complete: Both Product and Issue are provided.'\n\n"
        f"Additionally, suggest the next step for the user to provide the missing details."
    )
    response = chat.predict(prompt, max_tokens=100).strip()
    return response

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Chat Input for Context Collection
if not st.session_state.context_collected:
    if user_input := st.chat_input("Describe your issue:"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Check for missing details
        product_categories = df1['Product'].unique()
        evaluation_response = check_missing_details(chat, user_input, product_categories)

        if "Complete" in evaluation_response:
            st.session_state.context_collected = True
            st.session_state.chat_history.append({"role": "assistant", "content": "Thank you! You have provided sufficient details. Click the 'Start Classification' button to begin."})
            st.chat_message("assistant").write("Thank you! You have provided sufficient details. Click the 'Start Classification' button to begin.")
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": evaluation_response})
            st.chat_message("assistant").write(evaluation_response)

# Button to Start Classification Process
if st.session_state.context_collected and not st.session_state.classification_started:
    if st.button("Start Classification"):
        st.session_state.classification_started = True

        # Classification Process
        try:
            user_input = st.session_state.chat_history[-1]["content"]

            # Classify by Product
            product_categories = df1['Product'].unique()
            response_product = chat.predict(
                f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
                f"Complaint: {user_input}. Respond with the exact product as written there."
            )
            assigned_product = response_product.strip()

            # Classify by Sub-product
            subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
            response_subproduct = chat.predict(
                f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
                f"Complaint: {user_input}. Respond with the exact sub-product as written there."
            )
            assigned_subproduct = response_subproduct.strip()

            # Classify by Issue
            issue_options = df1[(df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)]['Issue'].unique()
            response_issue = chat.predict(
                f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
                f"Complaint: {user_input}. Respond with the exact issue as written there."
            )
            assigned_issue = response_issue.strip()

            # Save results to session state
            st.session_state.assigned_product = assigned_product
            st.session_state.assigned_subproduct = assigned_subproduct
            st.session_state.assigned_issue = assigned_issue

            # Display Classification Results
            classification_summary = (
                f"Classification Results:\n"
                f"- **Product**: {assigned_product}\n"
                f"- **Sub-product**: {assigned_subproduct}\n"
                f"- **Issue**: {assigned_issue}"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": classification_summary})
            st.chat_message("assistant").write(classification_summary)

        except Exception as e:
            error_message = f"Error during classification: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.get('assigned_product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.get('assigned_subproduct', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.get('assigned_issue', 'N/A')}")
