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
st.write("A chatbot to classify customer complaints and create Jira tasks if needed.")

# Initialize Session State
if "memory" not in st.session_state:
    # Initialize memory with conversation history
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)

if "classification_started" not in st.session_state:
    st.session_state.classification_started = False

if "product_described" not in st.session_state:
    st.session_state.product_described = False

if "problem_described" not in st.session_state:
    st.session_state.problem_described = False

if "classification_results" not in st.session_state:
    st.session_state.classification_results = {}

# Load Dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

product_categories = df1['Product'].unique()

# Initialize OpenAI Chat
try:
    model_type = "gpt-4o-mini"
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model=model_type)
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Helper function to analyze input details
def evaluate_input_details(chat, user_input, memory_messages):
    """
    Analyze user input to identify missing details using memory context.
    """
    prompt = (
        f"You are a helpful assistant analyzing customer complaints. Based on the conversation so far:\n"
        f"{memory_messages}\n\n"
        f"The user just mentioned:\n'{user_input}'\n\n"
        f"Your task is to determine if the user has provided:\n"
        f"1. A product (e.g., credit card, savings account).\n"
        f"2. A specific issue or problem (e.g., 'fraudulent transactions', 'stolen card').\n\n"
        f"Respond naturally and warmly to acknowledge provided details and politely ask for any missing information, "
        f"only related to product and issue. Finish collecting information when you have sufficient details for classification."
    )
    return chat.predict(prompt).strip()

# Helper function for classification
def classify_complaint(chat, prompt):
    response = chat.predict(prompt).strip()
    return response

# Display Chat History
st.write("### Chat History")
for message in st.session_state.memory.chat_memory.messages:
    st.chat_message(message["role"]).write(message["content"])

if user_input := st.chat_input("Describe your issue:"):
    st.session_state.memory.chat_memory.add_message({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get memory messages for context
    memory_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.chat_memory.messages])

    # Evaluate Input Details
    evaluation_response = evaluate_input_details(chat, user_input, memory_messages)
    st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": evaluation_response})
    st.chat_message("assistant").write(evaluation_response)

    # Update flags based on response
    if "product" in evaluation_response.lower():
        st.session_state.product_described = True
    if "issue" in evaluation_response.lower():
        st.session_state.problem_described = True

# Classification Process Trigger
if st.session_state.product_described and st.session_state.problem_described:
    if not st.session_state.classification_started:
        if st.button("Start Classification"):
            st.session_state.classification_started = True

            # Classification Process
            try:
                memory_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.chat_memory.messages])

                # Step 1: Classify by Product
                product_prompt = (
                    f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
                    f"Complaint details: {memory_messages}. Respond with the exact product as written there."
                )
                assigned_product = classify_complaint(chat, product_prompt)

                # Step 2: Classify by Sub-product
                subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
                subproduct_prompt = (
                    f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
                    f"Complaint details: {memory_messages}. Respond with the exact sub-product as written there."
                )
                assigned_subproduct = classify_complaint(chat, subproduct_prompt)

                # Step 3: Classify by Issue
                issue_options = df1[
                    (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
                ]['Issue'].unique()
                issue_prompt = (
                    f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
                    f"Complaint details: {memory_messages}. Respond with the exact issue as written there."
                )
                assigned_issue = classify_complaint(chat, issue_prompt)

                # Save Results
                st.session_state.classification_results = {
                    "Product": assigned_product,
                    "Sub-product": assigned_subproduct,
                    "Issue": assigned_issue,
                }

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

# Summary Button
if st.button("Show Classification Summary"):
    results = st.session_state.classification_results
    st.write("### Classification Summary")
    st.write(f"- **Product**: {results.get('Product', 'N/A')}")
    st.write(f"- **Sub-product**: {results.get('Sub-product', 'N/A')}")
    st.write(f"- **Issue**: {results.get('Issue', 'N/A')}")
