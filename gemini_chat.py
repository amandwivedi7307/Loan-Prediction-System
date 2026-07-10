import streamlit as st
from google import genai
import time

# -----------------------------
# Gemini Client
# -----------------------------
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def loan_chatbot(user_data):

    st.markdown("## 🤖 AI Loan Assistant")

    # Chat History
    # -----------------------------
# Conversation
# -----------------------------
    if st.session_state["messages"]:

        st.markdown("### 💬 Conversation")

        for msg in st.session_state["messages"]:

            if msg["role"] == "user":
                st.info("👤 " + msg["content"])

            else:
                st.success("🤖 " + msg["content"])

    prompt = st.text_area(
        "💬 Ask AI",
        placeholder="Example: Why was my loan rejected?",
        height=120,
        key="loan_ai_prompt"
    )

    ask = st.button(
        "🚀 Ask Gemini",
        use_container_width=True,
        key="loan_ai_button"
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

Question:
{prompt}

Answer in simple English.
"""

        with st.spinner("🤖 Thinking..."):

            answer = None

            for model_name in [
                "gemini-flash-latest",
                "gemini-2.0-flash"
            ]:

                try:

                    response = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt
                    )

                    answer = response.text
                    break

                except Exception as e:

                    if "503" in str(e):
                        time.sleep(3)
                        st.warning("⚠️ Gemini is busy. Retrying...")
                        continue
                    elif "429" in str(e):
                        st.warning("⚠️ AI quota reached. Please try again after a minute.")
                        continue
                    else:
                        answer = str(e)
                        break

        if answer is None:
            answer = "Gemini server is busy. Please try again."

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        if st.session_state["messages"]:
            st.markdown("### Conversation")

            for msg in st.session_state["messages"]:

                if msg["role"] == "user":
                    st.info("👤 " + msg["content"])

                else:
                    st.success("🤖 " + msg["content"])
        

        