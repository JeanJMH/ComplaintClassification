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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "product" not in st.session_state:
    st.session_state.product = None
if "problem" not in st.session_state:
    st.session_state.problem = None
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

# Helper function to identify missing details
def identify_missing_details(input_text, product_categories):
    product_match = any(product.lower() in input_text.lower() for product in product_categories)
    problem_keywords = ["fraud", "stolen", "issue", "problem", "lost", "dispute", "charge", "unauthorized"]
    problem_match = any(keyword in input_text.lower() for keyword in problem_keywords)
    return product_match, problem_match

# Display Chat History
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# Chat Input and Workflow
if user_input := st.chat_input("Describe your issue:"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    product_categories = df1['Product'].unique()
    product_match, problem_match = identify_missing_details(user_input, product_categories)

    # Update session state based on input
    if not st.session_state.product and product_match:
        for product in product_categories:
            if product.lower() in user_input.lower():
                st.session_state.product = product
                break

    if not st.session_state.problem and problem_match:
        st.session_state.problem = user_input

    # Check missing details
    if not st.session_state.product or not st.session_state.problem:
        missing_details = []
        if not st.session_state.product:
            missing_details.append("product (e.g., 'Credit Card', 'Savings Account')")
        if not st.session_state.problem:
            missing_details.append("problem description (e.g., 'stolen', 'fraudulent transactions')")

        query_message = f"It seems your input lacks sufficient details. Could you specify the missing information: {', '.join(missing_details)}?"
        st.session_state.chat_history.append({"role": "assistant", "content": query_message})
        st.chat_message("assistant").write(query_message)
    else:
        if not st.session_state.classification_started:
            st.session_state.classification_started = True
            st.session_state.chat_history.append({"role": "assistant", "content": "Thank you! Starting the classification process..."})
            st.chat_message("assistant").write("Thank you! Starting the classification process...")

            # Classification Process
            try:
                # Step 1: Classify by Product
                response_product = chat.predict(
                    f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
                    f"Complaint: {st.session_state.problem}. Respond with the exact product as written there."
                )
                assigned_product = response_product.strip()

                # Step 2: Classify by Sub-product
                subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
                response_subproduct = chat.predict(
                    f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
                    f"Complaint: {st.session_state.problem}. Respond with the exact sub-product as written there."
                )
                assigned_subproduct = response_subproduct.strip()

                # Step 3: Classify by Issue
                issue_options = df1[(df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)]['Issue'].unique()
                response_issue = chat.predict(
                    f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
                    f"Complaint: {st.session_state.problem}. Respond with the exact issue as written there."
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

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.get('assigned_product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.get('assigned_subproduct', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.get('assigned_issue', 'N/A')}")
Improvements:
Dynamic Interaction:

The chatbot only asks for missing information, ensuring a seamless user experience.
Once all details are available, it proceeds to the classification process.
State Preservation:

Stores product and problem details in session state, avoiding redundant questions.
Automatically updates the state with provided information.
Error Handling:

Classification errors are shown as assistant messages, not as errors in the interface.
Classification Results:

Results are presented as part of the conversation and stored for summary viewing.
Let me know if more adjustments are needed!






You said:
dont use this use a prompt and the model to identify if there are enough information, if there are a product and issue in the chat: def identify_missing_details(input_text, product_categories):
    product_match = any(product.lower() in input_text.lower() for product in product_categories)
    problem_keywords = ["fraud", "stolen", "issue", "problem", "lost", "dispute", "charge", "unauthorized"]
    problem_match = any(keyword in input_text.lower() for keyword in problem_keywords)
    return product_match, problem_match
ChatGPT said:
ChatGPT
To leverage a prompt and the model to determine if the user input contains enough information (e.g., a product and issue) instead of using keyword-based logic, we can use the following approach. The model will evaluate the input text directly and decide whether sufficient details are provided.

Full Code Using Prompt-Based Evaluation
python
Copy code
import streamlit as st
from langchain_openai import ChatOpenAI
import pandas as pd

# Title and Description
st.title("💬 Financial Complaint Classifier")
st.write("A chatbot to classify customer complaints and create Jira tasks if needed.")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "product" not in st.session_state:
    st.session_state.product = None
if "problem" not in st.session_state:
    st.session_state.problem = None
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

# Helper function to evaluate sufficient details using the model
def evaluate_input_details(chat, user_input, product_categories):
    prompt = (
        f"You are an assistant helping classify customer complaints. Determine if the following input contains enough details to classify:\n\n"
        f"Input: '{user_input}'\n\n"
        f"Details needed:\n"
        f"1. A product (e.g., {', '.join(product_categories)})\n"
        f"2. A clear problem or issue (e.g., 'stolen', 'fraudulent transactions', 'unauthorized charges').\n\n"
        f"Respond with one of the following:\n"
        f"- 'Sufficient details: [Product] and [Issue] identified.'\n"
        f"- 'Missing: [Specify what is missing, e.g., Product or Issue]'\n"
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

    if "Sufficient details" in evaluation_response:
        if not st.session_state.classification_started:
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
