from app.inference import infer
import streamlit as st
st.title("🤖 Persona Chatbot")
st.markdown("*A conversational AI fine-tuned on the Persona-Chat dataset*")
with st.expander("ℹ️ About this project"):
    st.write("""
    Persona Chatbot is a conversational AI model fine-tuned
    on the Persona-Chat dataset.

    The model generates responses based on conversational
    context and learned persona characteristics.

    ⚠️ Limitations: This is a portfolio project built using 
    a fine-tuned language model. The assistant may sometimes hallucinate,
    generate irrelevant responses, or lose conversational context.
    """)
if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
user_input=st.chat_input("Type anything...")
if user_input:
    st.session_state.messages.append({
        "role":"User",
        "content":user_input
    })
    answer=infer(user_input)
    st.session_state.messages.append({
        "role":"Assistant",
        "content":answer
    })
    st.rerun()
with st.sidebar:
    st.header("Persona Chatbot")
    button=st.button("Clear Chat")
    if button:
        st.session_state.messages=[]
        st.rerun()
    st.markdown("""
    **Base model:** GPT-2  
    **Dataset:** Persona-Chat  
    **Framework:** Transformers
    """)