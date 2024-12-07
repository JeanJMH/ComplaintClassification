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
def evaluate_input_details(chat, user_input, memory_context):
    """
    Analyze user input to identify missing details using memory context.
    """
    prompt = (
        f"You are a helpful assistant analyzing customer complaints. Based on the conversation so far:\n"
        f"{memory_context}\n\n"
        f"The user just mentioned:\n'{user_input}'\n\n"
        f"Your task is to determine if the user has provided:\n"
        f"1. A product (e.g., credit card, savings account).\n"
        f"2. A specific issue or problem (e.g., 'fraudulent transactions', 'stolen card').\n\n"
        f"Respond naturally and warmly to acknowledge provided details, and politely ask for any missing information. "
        f"Be concise but empathetic in your responses."
    )
    return chat.predict(prompt).strip()

# Chat Input and Workflow
st.write("### Chat History")
for message in st.session_state.memory.chat_memory.messages:
    st.chat_message(message.role).write(message.content)

if user_input := st.chat_input("Describe your issue:"):
    st.session_state.memory.chat_memory.add_message({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Evaluate Input Details
    memory_context = st.session_state.memory.chat_memory.get_buffer()
    evaluation_response = evaluate_input_details(chat, user_input, memory_context)
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
                memory_context = st.session_state.memory.chat_memory.get_buffer()

                # Step 1: Classify by Product
                response_product = chat.predict(
                    f"Based on this conversation: {memory_context}\n"
                    f"Classify the complaint by matching it to one of these Product categories: {product_categories.tolist()}."
                )
                assigned_product = response_product.strip()

                # Step 2: Classify by Sub-product
                subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
                response_subproduct = chat.predict(
                    f"Based on this conversation: {memory_context}\n"
                    f"Classify the complaint into one of these Sub-product categories under '{assigned_product}': {subproduct_options.tolist()}."
                )
                assigned_subproduct = response_subproduct.strip()

                # Step 3: Classify by Issue
                issue_options = df1[
                    (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
                ]['Issue'].unique()
                response_issue = chat.predict(
                    f"Based on this conversation: {memory_context}\n"
                    f"Classify the complaint into one of these Issue categories under '{assigned_product}' and '{assigned_subproduct}': {issue_options.tolist()}."
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

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.get('assigned_product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.get('assigned_subproduct', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.get('assigned_issue', 'N/A')}")
