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
st.write("Hi! I’m here to assist you in classifying your financial complaint. Let's get started!")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "context_ready" not in st.session_state:
    st.session_state.context_ready = False
if "classification_started" not in st.session_state:
    st.session_state.classification_started = False
if "context" not in st.session_state:
    st.session_state.context = {"product": None, "issue": None}

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
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Helper function to evaluate and collect missing context
def collect_context(chat, user_input, context):
    """
    Use GPT to collect missing context and update the session state.
    """
    prompt = (
        f"You are a friendly assistant helping users classify financial complaints. Start each response with a warm greeting and explain that you're here to help them with their issue. "
        f"Review the following user input and provide feedback about missing details or confirm that all details are complete. "
        f"The user's input so far:\n\n"
        f"Product: {context['product'] or 'None'}\n"
        f"Issue: {context['issue'] or 'None'}\n\n"
        f"The latest user input is: '{user_input}'\n\n"
        f"Explain what is missing in a clear and polite way. Respond with:\n"
        f"- A greeting and a request for the missing product or issue details (e.g., 'Could you let me know which product you're referring to, such as a credit card or mortgage?')\n"
        f"- If all information is complete: A thank-you message confirming all details and mentioning that the support team will handle the case."
    )
    response = chat.predict(prompt).strip()
    return response

# Chat Input for Context Collection
if not st.session_state.classification_started:
    if user_input := st.chat_input("Describe your issue:"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Use GPT to collect missing context
        response = collect_context(chat, user_input, st.session_state.context)

        # Check if details are complete
        if "support team will handle the case" in response.lower():
            st.session_state.context_ready = True
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

            # Mark conversation as complete
            st.session_state.classification_started = True
        else:
            # Update context dynamically
            if "product" in response.lower() and st.session_state.context["product"] is None:
                st.session_state.context["product"] = user_input
            if "issue" in response.lower() and st.session_state.context["issue"] is None:
                st.session_state.context["issue"] = user_input

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

# Classification Process (Triggered by Button)
if st.session_state.context_ready:
    # Provide a final thank-you message after the classification button is clicked.
    if st.button("Start Classification"):
        st.session_state.chat_history.append({"role": "assistant", "content": "Starting the classification process..."})
        st.chat_message("assistant").write("Starting the classification process...")

        try:
            # Step 1: Classify by Product
            product_categories = df1['Product'].unique()
            response_product = chat.predict(
                f"You are a financial expert classifying complaints. Classify the following complaint based on these Product categories: {product_categories.tolist()}.\n\nComplaint: {st.session_state.context['product']}. Respond with the exact product as written there."
            )
            assigned_product = response_product.strip()

            # Step 2: Classify by Sub-product
            subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
            response_subproduct = chat.predict(
                f"Classify the complaint based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}.\n\nComplaint: {st.session_state.context['issue']}. Respond with the exact sub-product as written there."
            )
            assigned_subproduct = response_subproduct.strip()

            # Step 3: Classify by Issue
            issue_options = df1[(df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)]['Issue'].unique()
            response_issue = chat.predict(
                f"Classify the complaint based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}.\n\nComplaint: {st.session_state.context['issue']}. Respond with the exact issue as written there."
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
                f"- **Issue**: {assigned_issue}\n\n"
                f"Thank you for providing the details! Our support team will contact you to resolve your case. Goodbye!"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": classification_summary})
            st.chat_message("assistant").write(classification_summary)

        except Exception as e:
            error_message = f"Error during classification: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)
