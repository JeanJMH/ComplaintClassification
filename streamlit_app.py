import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# Title and Description
st.title("💬 Financial Complaint Classifier")
st.write("Classify customer complaints into Product, Sub-product, and Issue categories.")

# Load classification dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Chat Input and Classification Workflow
if client_complaint := st.chat_input("Describe your issue:"):
    st.chat_message("user").write(client_complaint)

    # Step 1: Classify by Product
    product_categories = df1['Product'].unique()
    response_product = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
                "Respond with the exact product as written there."
            )},
            {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
        ],
        max_tokens=20,
        temperature=0.1
    )
    assigned_product = response_product.choices[0].message.content.strip()
    st.write(f"Assigned Product: {assigned_product}")

    # Step 2: Classify by Sub-product
    subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
    response_subproduct = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
                "Respond with the exact sub-product as written there."
            )},
            {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
        ],
        max_tokens=20,
        temperature=0.1
    )
    assigned_subproduct = response_subproduct.choices[0].message.content.strip()
    st.write(f"Assigned Sub-product: {assigned_subproduct}")

    # Step 3: Classify by Issue
    issue_options = df1[
        (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
    ]['Issue'].unique()
    response_issue = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
                "Respond with the exact issue as written there."
            )},
            {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
        ],
        max_tokens=20,
        temperature=0.1
    )
    assigned_issue = response_issue.choices[0].message.content.strip()
    st.write(f"Assigned Issue: {assigned_issue}")

    # Display Classification Results
    st.chat_message("assistant").write(
        f"Classification Results:\n- **Product**: {assigned_product}\n- **Sub-product**: {assigned_subproduct}\n- **Issue**: {assigned_issue}"
    )

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {assigned_product}")
    st.write(f"- **Sub-product**: {assigned_subproduct}")
    st.write(f"- **Issue**: {assigned_issue}")
