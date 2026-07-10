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
    st.session_state["messages"] = []


# -----------------------------
# Chat Function
# -----------------------------

def loan_chatbot(user_data):

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.write("---")
    st.subheader("🤖 AI Loan Assistant")

    # Show old messages
    for msg in st.session_state["messages"]:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    prompt = st.text_area(
        "💬 Ask AI",
        placeholder="Example: Why was my loan rejected?",
        height=120,
        key="ai_question"
    )

    ask = st.button(
        "🤖 Ask Gemini",
        use_container_width=True,
        key="ask_gemini"
    )

    if ask:
        if prompt.strip() == "":
            st.warning("Please enter a question.")
            return

        st.session_state["messages"].append(
    {
        "role": "user",
        "content": prompt
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

                    import time

                    answer = None

                    models = [
                        "gemini-flash-latest",
                        "gemini-2.5-flash"
                    ]

                    for model_name in models:

                        try:

                            response = client.models.generate_content(
                                model=model_name,
                                contents=full_prompt
                            )

                            answer = response.text

                            break

                        except Exception as e:

                            if "503" in str(e):

                                time.sleep(5)

                            else:

                                answer = str(e)

                    if answer:
                        st.markdown(answer)
                    else:
                        st.error("Gemini server is busy. Please try again.")

                    answer = response.text

                except Exception as e:

                    answer = f"Error : {e}"

                st.markdown(answer)

                st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer
                }
                )