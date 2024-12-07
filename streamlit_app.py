import streamlit as st
import pandas as pd
import os
from datetime import date
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain import hub

# Title and Description
st.title("💬 Financial Support Chatbot")
st.write("Classify client complaints into product, subproduct, and issue categories, and create Jira tasks for tracking.")

# Load Classification Dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Extract Product Categories
product_categories = df1['Product'].unique()

# Initialize Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=10, return_messages=True
    )
    st.session_state.problem_described = False
    st.session_state.product_described = None
    st.session_state.jira_task_created = False

# Helper Functions for Classification
def classify_input(input_text, options, context):
    """
    Classify the input_text based on the provided options using OpenAI.
    """
    client = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")
    response = client.completions.create(
        messages=[
            {"role": "system", "content": f"You are an AI assistant that classifies complaints into {context}: {options}. Respond with the exact category."},
            {"role": "user", "content": input_text},
        ],
        max_tokens=20,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# Handle User Input and Classification
if user_input := st.chat_input("Describe your issue:"):
    st.chat_message("user").write(user_input)

    # Classify by Product
    product = classify_input(user_input, product_categories.tolist(), "Product Categories")
    
    # Classify by Sub-product
    subproduct_options = df1[df1['Product'] == product]['Sub-product'].unique()
    subproduct = classify_input(user_input, subproduct_options.tolist(), f"Sub-product Categories for '{product}'")

    # Classify by Issue
    issue_options = df1[(df1['Product'] == product) & (df1['Sub-product'] == subproduct)]['Issue'].unique()
    issue = classify_input(user_input, issue_options.tolist(), f"Issues for '{product}' and '{subproduct}'")

    # Display Classification Results
    st.chat_message("assistant").write(
        f"Complaint classified into:\n- **Product**: {product}\n- **Sub-product**: {subproduct}\n- **Issue**: {issue}"
    )

    # Store Classification in Session State
    st.session_state.problem_described = True
    st.session_state.product_described = product
    st.session_state.subproduct_described = subproduct
    st.session_state.issue_described = issue

# Jira Task Creation
if st.session_state.problem_described and st.session_state.product_described and not st.session_state.jira_task_created:
    st.write("Starting Jira task creation process...")
    try:
        # Setup Jira API Credentials
        os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
        os.environ["JIRA_USERNAME"] = "jeanmh@bu.edu"
        os.environ["JIRA_INSTANCE_URL"] = "https://jmhu.atlassian.net"
        
        # Extract User Description
        user_description = user_input

        # Define Jira Task Details
        assigned_issue = f"Managing my {st.session_state.product_described} Account"
        question = (
            f"Create a task in my project with the key LLMTS. The task's type is 'Task', assigned to rich@bu.edu. "
            f"The summary is '{assigned_issue}'. "
            f"Always assign 'Highest' priority if the issue is related to fraudulent activities. "
            f"Use 'High' priority for other issues. "
            f"The description is '{user_description}'"
        )

        # Initialize Jira Toolkit and Agent
        jira = JiraAPIWrapper()
        toolkit = JiraToolkit.from_jira_api_wrapper(jira)

        # Modify Toolkit for Proper Names
        for idx, tool in enumerate(toolkit.tools):
            toolkit.tools[idx].name = toolkit.tools[idx].name.replace(" ", "_")
            if "create_issue" in toolkit.tools[idx].name:
                toolkit.tools[idx].description += " Ensure to specify the project ID."

        # Create Jira Task
        tools = toolkit.get_tools()
        chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")
        prompt = hub.pull("hwchase17/react")
        agent = create_tool_calling_agent(chat, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        result = agent_executor.invoke({"input": question})
        st.success(f"Jira task created successfully: {result}")
        st.session_state.jira_task_created = True
    except Exception as e:
        st.error(f"Error during Jira task creation: {e}")

# Display Summary
if st.button("Show Summary"):
    if "product_described" in st.session_state and "subproduct_described" in st.session_state and "issue_described" in st.session_state:
        st.write("### Summary of Classification and Task")
        st.write(f"- **Product**: {st.session_state.product_described}")
        st.write(f"- **Sub-product**: {st.session_state.subproduct_described}")
        st.write(f"- **Issue**: {st.session_state.issue_described}")
        st.write("---")
