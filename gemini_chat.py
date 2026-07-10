import streamlit as st
from groq import Groq
import time

# -----------------------------
# Gemini Client
# -----------------------------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []




def loan_chatbot(user_data):

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.markdown("## 🤖 AI Loan Assistant")

    

    # Chat History
    # -----------------------------
# Conversation
# -----------------------------
    if "messages" in st.session_state and st.session_state["messages"]:

        st.markdown("### 💬 Conversation")

        for msg in st.session_state["messages"]:

            if msg["role"] == "user":
                st.info("👤 " + msg["content"])

            else:
                st.success("🤖 " + msg["content"])

    with st.form("loan_chat_form", clear_on_submit=True):

        prompt = st.text_area(
            "💬 Ask AI",
            placeholder="Example: Why was my loan rejected?",
            height=120
        )

        ask = st.form_submit_button(
            "🚀 Ask AI",
            use_container_width=True
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

        with st.spinner("🤖 AI is Thinking..."):
            answer = None

            try:

                response = client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=[

                        {
                            "role": "system",
                            "content": "You are a professional AI Loan Advisor. Give short, clear and practical answers."
                        },

                        {
                            "role": "user",
                            "content": full_prompt
                        }

                    ],

                    temperature=0.5,
                    max_tokens=500

                )

                answer = response.choices[0].message.content

            except Exception as e:

                answer = f"❌ Error : {e}"

            

        if answer is None:
            answer = "Groq server is busy. Please try again."

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        st.rerun()


        
        

        