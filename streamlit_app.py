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


st.title("💬 Financial Complaint Classifier with Jira Integration and Conversation Tracking")
st.write("Interact with the chatbot to classify complaints and manage Jira tasks.")

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

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

# Initialize Jira
try:
    os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
    os.environ["JIRA_USERNAME"] = "jeanmh@bu.edu"
    os.environ["JIRA_INSTANCE_URL"] = "https://jmhu.atlassian.net"
    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    tools = toolkit.get_tools()
except KeyError:
    st.warning("Jira API keys are missing. Jira task creation will be disabled.")

# Helper function to create a Jira task
def create_jira_task(product, sub_product, issue, user_description):
    try:
        task_summary = f"Complaint about {product} - {sub_product}"
        task_description = (
            f"Issue: {issue}\n"
            f"User Description: {user_description}"
        )
        task_priority = "High" if "fraud" in user_description.lower() else "Medium"
        question = (
            f"Create a Jira task with the following details:\n"
            f"Summary: {task_summary}\n"
            f"Description: {task_description}\n"
            f"Priority: {task_priority}\n"
            f"Project: LLMTS"
        )
        agent = toolkit.create_agent(chat, tools)
        result = agent.invoke({"input": question})
        return result
    except Exception as e:
        return f"Error creating Jira task: {e}"

# Chat Input and Classification Workflow
if user_input := st.chat_input("Describe your issue or continue the conversation:"):
    # Add user message to memory
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.chat_message("user").write(user_input)

    try:
        # Step 1: Classify by Product
        product_categories = df1['Product'].unique()
        product_prompt = (
            f"You are a financial expert assisting with complaints. Classify the issue into one of these product categories: {product_categories.tolist()}."
        )
        response_product = chat.predict_messages(
            [{"role": "system", "content": product_prompt}, {"role": "user", "content": user_input}]
        )
        assigned_product = response_product.content.strip()

        # Step 2: Classify by Sub-product
        subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
        subproduct_prompt = (
            f"You are a financial expert. Classify the complaint into one of these sub-product categories under '{assigned_product}': {subproduct_options.tolist()}."
        )
        response_subproduct = chat.predict_messages(
            [{"role": "system", "content": subproduct_prompt}, {"role": "user", "content": user_input}]
        )
        assigned_subproduct = response_subproduct.content.strip()

        # Step 3: Classify by Issue
        issue_options = df1[
            (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
        ]['Issue'].unique()
        issue_prompt = (
            f"You are a financial expert. Classify the complaint into one of these issue categories under '{assigned_product}' and '{assigned_subproduct}': {issue_options.tolist()}."
        )
        response_issue = chat.predict_messages(
            [{"role": "system", "content": issue_prompt}, {"role": "user", "content": user_input}]
        )
        assigned_issue = response_issue.content.strip()

        # Add assistant response to memory
        assistant_response = (
            f"Classification Results:\n- **Product**: {assigned_product}\n- **Sub-product**: {assigned_subproduct}\n- **Issue**: {assigned_issue}"
        )
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        st.chat_message("assistant").write(assistant_response)

        # Optional: Create Jira Task
        if st.button("Create Jira Task"):
            jira_result = create_jira_task(assigned_product, assigned_subproduct, assigned_issue, user_input)
            if "Error" in jira_result:
                st.error(jira_result)
            else:
                st.success(f"Jira Task Created: {jira_result}")

    except Exception as e:
        st.error(f"Error during classification: {e}")
        st.chat_message("assistant").write("Could not classify the product. Please refine your input.")

# Display Conversation History
st.write("### Conversation History")
for message in st.session_state.memory.chat_memory.messages:
    if message.type == "user":
        st.chat_message("user").write(message.content)
    elif message.type == "assistant":
        st.chat_message("assistant").write(message.content)
