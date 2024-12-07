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
if "context" not in st.session_state:
    st.session_state.context = {"product": None, "issue": None}
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
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Helper function to collect context
def collect_context(chat, user_input, context):
    """
    Use GPT to collect missing context and update the session state dynamically.
    """
    # Identify what is missing in the context
    missing_parts = []
    if not context["product"]:
        missing_parts.append("product (e.g., 'credit card', 'savings account')")
    if not context["issue"]:
        missing_parts.append("issue (e.g., 'fraudulent transactions', 'stolen card')")

    # Generate a dynamic prompt based on missing parts
    if missing_parts:
        prompt = (
            f"You are a friendly assistant helping users provide details for their financial complaints. Start with a warm greeting. "
            f"The user has shared the following information so far:\n\n"
            f"Product: {context['product'] or 'Not provided'}\n"
            f"Issue: {context['issue'] or 'Not provided'}\n\n"
            f"The latest user input is: '{user_input}'\n\n"
            f"Kindly request the missing information: {', '.join(missing_parts)}. "
            f"Ensure your response is polite, encouraging, and helpful."
        )
    else:
        prompt = (
            f"You are a friendly assistant confirming the user's details for a financial complaint. Start with a warm thank-you message. "
            f"The provided details are:\n\n"
            f"Product: {context['product']}\n"
            f"Issue: {context['issue']}\n\n"
            f"Reassure the user that the details are sufficient and inform them that you will proceed with the classification."
        )

    # Generate the response
    response = chat.predict(prompt).strip()

    # Update context based on user input
    if not context["product"] and "credit card" in user_input.lower():  # Example product detection
        context["product"] = "Credit Card"
    if not context["issue"] and "stolen" in user_input.lower():  # Example issue detection
        context["issue"] = "Stolen Money"

    return response

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Chat Input and Workflow
if user_input := st.chat_input("Describe your issue:"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Collect Context
    context_response = collect_context(chat, user_input, st.session_state.context)

    # If context is complete, start classification
    if all(st.session_state.context.values()):
        st.session_state.classification_started = True
        st.session_state.chat_history.append({"role": "assistant", "content": "Thank you! Starting the classification process..."})
        st.chat_message("assistant").write("Thank you! Starting the classification process...")

        # Classification Process
        try:
            # Step 1: Classify by Product
            product_categories = df1['Product'].unique()
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
        st.session_state.chat_history.append({"role": "assistant", "content": context_response})
        st.chat_message("assistant").write(context_response)

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.get('assigned_product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.get('assigned_subproduct', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.get('assigned_issue', 'N/A')}")
