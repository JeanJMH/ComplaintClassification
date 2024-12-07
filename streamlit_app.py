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
st.title("💬 Financial Complaint Classifier & Task Manager")
st.write("Classify customer complaints into Product, Sub-product, and Issue categories, and automatically create Jira tasks.")

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

# Initialize Jira API
try:
    os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
    os.environ["JIRA_USERNAME"] = "jeanmh@bu.edu"
    os.environ["JIRA_INSTANCE_URL"] = "https://jmhu.atlassian.net"
    os.environ["JIRA_CLOUD"] = "True"
    jira = JiraAPIWrapper()
    jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira)
except KeyError:
    st.error("Jira API credentials missing! Please set 'JIRA_API_TOKEN', 'JIRA_USERNAME', and 'JIRA_INSTANCE_URL' in your Streamlit secrets.")
    st.stop()

# Helper function for classification
def classify_complaint(chat, prompt, user_input):
    """
    Classify complaints into categories (Product, Sub-product, or Issue) using the ChatOpenAI model.
    """
    try:
        response = chat.predict(
            prompt=f"{prompt}\nUser complaint: '{user_input}'",
        )
        return response.strip()
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

# Helper function to create Jira task
def create_jira_task(summary, description, priority="High"):
    """
    Create a Jira task using the Jira API.
    """
    try:
        task_details = {
            "project": {"key": "LLMTS"},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"},
            "priority": {"name": priority},
        }
        task = jira_toolkit.jira_api.create_issue(fields=task_details)
        return task.key
    except Exception as e:
        st.error(f"Error during Jira task creation: {e}")
        return None

# Chat Input and Classification Workflow
if client_complaint := st.chat_input("Describe your issue:"):
    st.chat_message("user").write(client_complaint)

    # Step 1: Classify by Product
    product_categories = df1['Product'].unique()
    product_prompt = (
        f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
        "Respond with the exact product as written there."
    )
    assigned_product = classify_complaint(chat, product_prompt, client_complaint)
    if assigned_product:
        st.write(f"Assigned Product: {assigned_product}")
    else:
        st.error("Could not classify the product. Please refine your input.")
        st.stop()

    # Step 2: Classify by Sub-product
    subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
    if len(subproduct_options) == 0:
        st.error(f"No sub-products found for the product '{assigned_product}'.")
        st.stop()

    subproduct_prompt = (
        f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
        "Respond with the exact sub-product as written there."
    )
    assigned_subproduct = classify_complaint(chat, subproduct_prompt, client_complaint)
    if assigned_subproduct:
        st.write(f"Assigned Sub-product: {assigned_subproduct}")
    else:
        st.error("Could not classify the sub-product. Please refine your input.")
        st.stop()

    # Step 3: Classify by Issue
    issue_options = df1[
        (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
    ]['Issue'].unique()
    if len(issue_options) == 0:
        st.error(f"No issues found for the product '{assigned_product}' and sub-product '{assigned_subproduct}'.")
        st.stop()

    issue_prompt = (
        f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
        "Respond with the exact issue as written there."
    )
    assigned_issue = classify_complaint(chat, issue_prompt, client_complaint)
    if assigned_issue:
        st.write(f"Assigned Issue: {assigned_issue}")
    else:
        st.error("Could not classify the issue. Please refine your input.")
        st.stop()

    # Display Classification Results
    st.chat_message("assistant").write(
        f"Classification Results:\n- **Product**: {assigned_product}\n- **Sub-product**: {assigned_subproduct}\n- **Issue**: {assigned_issue}"
    )

    # Create Jira Task
    st.write("Creating Jira Task...")
    task_summary = f"Complaint about {assigned_product} ({assigned_subproduct})"
    task_description = f"User complaint: {client_complaint}\nClassified as:\n- Product: {assigned_product}\n- Sub-product: {assigned_subproduct}\n- Issue: {assigned_issue}"
    priority = "Highest" if "fraud" in assigned_issue.lower() else "High"

    jira_task_key = create_jira_task(task_summary, task_description, priority)
    if jira_task_key:
        st.success(f"Jira Task Created Successfully! Task Key: {jira_task_key}")
    else:
        st.error("Failed to create Jira task. Please try again later.")

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {assigned_product}")
    st.write(f"- **Sub-product**: {assigned_subproduct}")
    st.write(f"- **Issue**: {assigned_issue}")
