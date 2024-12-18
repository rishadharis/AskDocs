import streamlit as st
from streamlit_chat import message
from backend.core import run_llm
from typing import Set


# Add sidebar with user info
with st.sidebar:
    st.title("Busto AskDocs")
    st.subheader("Ask technical documentation with AI")
    
    st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp", width=100)
    st.write(f"👤 **Created by:** Rishad HB")
    st.write(f"📧 **Email:** hi@busto.dev")
    
    # Add divider
    st.divider()
    st.button("Request for new documentation", use_container_width=True, type="primary")

# Main content
# Add documentation selector
documentation = st.selectbox(
    "Select documentation",
    ["Langchain Documentation - (https://python.langchain.com/docs/introduction/)", "AWS Boto3 Documentation - (https://boto3.amazonaws.com/v1/documentation/api/latest/index.htmlhttps://boto3.amazonaws.com/v1/documentation/api/latest/)"]
)

# Create a form for the input
with st.form(key='prompt_form', clear_on_submit=True):
    prompt = st.text_input("Question", placeholder="Type your question here and press enter...")
    submit_button = st.form_submit_button("Send")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

def create_sources_links(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"[Source ({i+1})]({source})\n"
    return sources_string

if submit_button and prompt:
    with st.spinner("Searching on the documentation..."):
        response = run_llm(query=prompt)
        sources = set([doc.metadata["source"] for doc in response["source_documents"]])

        formatted_response = (
            f"{response['result']} \n\n {create_sources_links(sources)}"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)