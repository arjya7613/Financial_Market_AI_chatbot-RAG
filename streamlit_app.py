import json
import requests
import streamlit as st

st.set_page_config(
    page_title="Financial Intelligence AI",
    page_icon="💹",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/query"

st.title("💹 Financial Market Intelligence Chatbot")

st.caption(
    "Institutional-grade Financial Analysis powered by Multi-Agent RAG"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
            Hello 👋
            I am your Financial Intelligence Assistant.
            """
        }

    ]

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.selectbox(
        "Response Mode",
        ["simple", "detailed"]
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """
                    Hello 👋
                    I am your Financial Intelligence Assistant.
                    How can I help you today?
                """
            }
        ]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask a financial question..."
)

if query:

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]
    if query.lower().strip() in greetings:
        greeting_response = """
            Hello 👋
            How can I help you today?
        """
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": greeting_response
            }
        )
        with st.chat_message("assistant"):
            st.markdown(greeting_response)
        st.stop()   

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": query,
                    "mode": mode
                },
                stream=True
            )

            if response.status_code != 200:
                st.error(
                    f"API Error: {response.status_code}"
                )
            else:
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data:"):
                            json_data = decoded.replace(
                                "data: ",
                                ""
                            )
                            data = json.loads(
                                json_data
                            )

                            if "final_answer" in data:
                                final_answer = data[
                                    "final_answer"
                                ]
                                full_response = f"""
                                        # 📊 Final Financial Report

                                        {final_answer}
                                    """

                                response_placeholder.markdown(
                                    full_response
                                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )
        except requests.exceptions.ConnectionError:
            st.error(
                """
                Could not connect to FastAPI backend.
                """
            )
        except Exception as e:
            st.error(str(e))