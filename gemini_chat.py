import streamlit as st
from google import genai


# -----------------------------
# Gemini Client
# -----------------------------

client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"])


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Chat Function
# -----------------------------

def loan_chatbot(user_data):

    st.write("---")

    st.subheader("🤖 AI Loan Assistant")

    # Show old messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    prompt = st.chat_input(
        "Ask anything about your loan..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        full_prompt = f"""
You are an AI Loan Advisor.

Applicant Income : {user_data["Applicant Income"]}

Coapplicant Income : {user_data["Coapplicant Income"]}

Loan Amount : {user_data["Loan Amount"]}

Loan Term : {user_data["Loan Term"]}

Credit History : {user_data["Credit History"]}

Education : {user_data["Education"]}

Property Area : {user_data["Property Area"]}

Self Employed : {user_data["Self Employed"]}

Prediction : {user_data["Prediction"]}

Approval Probability : {user_data["Approval Probability"]:.2f}%

User Question:

{prompt}

Answer politely in simple English.
"""

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = client.models.generate_content(

                        model="gemini-3.5-flash",

                        contents=full_prompt

                    )

                    answer = response.text

                except Exception as e:

                    answer = f"Error : {e}"

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role":"assistant",
                        "content":answer
                    }
                )