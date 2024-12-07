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
st.write("Classify customer complaints into Product, Sub-product, and Issue categories and create Jira tasks.")

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

# Helper function for classification
def classify_complaint(chat, system_prompt, user_input):
    """
    Classify complaints into categories (Product, Sub-product, or Issue) using the ChatOpenAI model.
    """
    try:
        # Using the messages format required by the ChatOpenAI model
        response = chat.predict_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ])
        return response.content.strip()
    except Exception as e:
        st.error(f"Error during classification: {e}")
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

    # Jira Task Creation
    if st.button("Create Jira Task"):
        try:
            # Setup Jira API credentials
            os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
            os.environ["JIRA_USERNAME"] = "jeanmh@bu.edu"
            os.environ["JIRA_INSTANCE_URL"] = "https://jmhu.atlassian.net"

            # Define Jira task details
            jira_description = f"Complaint: {client_complaint}\nProduct: {assigned_product}\nSub-product: {assigned_subproduct}\nIssue: {assigned_issue}"
            summary = f"Issue with {assigned_product} - {assigned_subproduct}"
            priority = "Highest" if "fraud" in client_complaint.lower() else "High"

            # Initialize Jira Toolkit
            jira = JiraAPIWrapper()
            toolkit = JiraToolkit.from_jira_api_wrapper(jira)
            tools = toolkit.get_tools()

            # Create Jira task
            response = jira.create_issue(
                project_key="LLMTS",
                summary=summary,
                description=jira_description,
                issuetype="Task",
                priority=priority
            )
            st.success(f"Jira task created successfully! Task ID: {response['key']}")
        except Exception as e:
            st.error(f"Error creating Jira task: {e}")
