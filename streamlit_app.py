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

# Helper function to evaluate sufficient details using OpenAI model
def evaluate_input_details(chat, user_input, product_categories):
    prompt = (
        f"You are a helpful assistant analyzing customer complaints. Your task is to determine if the following input contains enough details to proceed:\n\n"
        f"Input: '{user_input}'\n\n"
        f"Check if the input mentions:\n"
        f"1. A product (e.g., {', '.join(product_categories)})\n"
        f"2. A clear problem or issue (e.g., 'stolen', 'fraudulent transactions', 'unauthorized charges').\n\n"
        f"Respond with one of the following:\n"
        f"- 'It seems your input is missing details about the product. Could you tell me which product you're referring to (e.g., Credit Card, Savings Account)?'\n"
        f"- 'It seems your input is missing details about the issue. Could you describe the problem you're experiencing (e.g., fraudulent transactions, stolen card)?'\n"
        f"- 'Your input seems complete. Let's proceed with the classification process.'"
    )
    response = chat.predict(prompt).strip()
    return response

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Chat Input and Workflow
if user_input := st.chat_input("Describe your issue:"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Evaluate Input Details
    product_categories = df1['Product'].unique()
    evaluation_response = evaluate_input_details(chat, user_input, product_categories)

    if "complete" in evaluation_response.lower():
        st.session_state.classification_started = True
        st.session_state.chat_history.append({"role": "assistant", "content": "Thank you! Starting the classification process..."})
        st.chat_message("assistant").write("Thank you! Starting the classification process...")

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
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": evaluation_response})
        st.chat_message("assistant").write(evaluation_response)

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.get('assigned_product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.get('assigned_subproduct', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.get('assigned_issue', 'N/A')}")
