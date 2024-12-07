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
st.write("Classify customer complaints into Product, Sub-product, and Issue categories and create Jira tasks if needed.")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "product_provided" not in st.session_state:
    st.session_state.product_provided = False
if "problem_provided" not in st.session_state:
    st.session_state.problem_provided = False

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

# Helper function for missing details evaluation
def identify_missing_details(input_text, product_categories):
    """
    Identify if product and problem details are missing from the input.
    """
    product_match = any(product.lower() in input_text.lower() for product in product_categories)
    problem_keywords = ["fraud", "stolen", "issue", "problem", "lost", "dispute", "charge", "unauthorized"]
    problem_match = any(keyword in input_text.lower() for keyword in problem_keywords)
    return product_match, problem_match

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Chat Input and Workflow
if client_complaint := st.chat_input("Describe your issue:"):
    # Add user's message to chat history
    st.session_state.chat_history.append({"role": "user", "content": client_complaint})
    st.chat_message("user").write(client_complaint)

    # Evaluate if product and problem details are provided
    product_categories = df1['Product'].unique()
    product_match, problem_match = identify_missing_details(client_complaint, product_categories)

    # Update session state based on input
    if product_match:
        st.session_state.product_provided = True
    if problem_match:
        st.session_state.problem_provided = True

    # Check if all required details are available
    if st.session_state.product_provided and st.session_state.problem_provided:
        st.session_state.chat_history.append({"role": "assistant", "content": "Thank you! Starting the classification process..."})
        st.chat_message("assistant").write("Thank you! Starting the classification process...")

        # Classification Process
        try:
            # Step 1: Classify by Product
            response_product = chat.predict(f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. Respond with the exact product.")
            assigned_product = response_product.strip()

            # Step 2: Classify by Sub-product
            subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
            response_subproduct = chat.predict(f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. Respond with the exact sub-product.")
            assigned_subproduct = response_subproduct.strip()

            # Step 3: Classify by Issue
            issue_options = df1[(df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)]['Issue'].unique()
            response_issue = chat.predict(f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. Respond with the exact issue.")
            assigned_issue = response_issue.strip()

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
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error during classification: {e}"})
            st.chat_message("assistant").write(f"Error during classification: {e}")

    else:
        missing_details = []
        if not st.session_state.product_provided:
            missing_details.append("product (e.g., 'Credit Card', 'Savings Account')")
        if not st.session_state.problem_provided:
            missing_details.append("problem description (e.g., 'stolen', 'fraudulent transactions')")

        # Respond with a dynamic query for missing details
        query_message = f"It seems your input lacks sufficient details. Could you specify the missing information: {', '.join(missing_details)}?"
        st.session_state.chat_history.append({"role": "assistant", "content": query_message})
        st.chat_message("assistant").write(query_message)

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.get('assigned_product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.get('assigned_subproduct', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.get('assigned_issue', 'N/A')}")
