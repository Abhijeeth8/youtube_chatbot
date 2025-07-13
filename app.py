import streamlit as st
from embedder import load_transcript, embed_transcript
from yt_chatbot import summarize_transcript, start_chatbot
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🎬 YouTube Video Chatbot")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "yt_transcript" not in st.session_state:
    st.session_state.yt_transcript = None


# Video input
video_url = st.text_input("🔗 Enter YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")

yt_transcript = load_transcript(video_url)
print("Embedding the video for the chatbot...")
vectorstore = embed_transcript(yt_transcript)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

col1, col2 = st.columns(2)

# Load transcript
if col1.button("Load Transcript"):
    try:
        st.session_state.yt_transcript = load_transcript(video_url)
        st.success("Transcript loaded.")
        # st.session_state.chat_history.append({"role": "ai", "content": "Transcript loaded successfully. You can now summarize or chat with the video."})
    except Exception as e:
        st.error(f"Error: {e}")

# Summarize
if col2.button("Summarize Transcript"):
    if st.session_state.yt_transcript:
        summary = summarize_transcript(st.session_state.yt_transcript)
        # st.session_state.chat_history.append({"role": "ai", "content": summary["ai"]})
        st.session_state.chat_history.append(AIMessage(summary["ai"]))
    else:
        st.warning("Please load a transcript first.")

# Chat interface
if st.session_state.yt_transcript:
    user_prompt = st.chat_input("Ask something about the video...")
    if user_prompt:
        st.session_state.chat_history.append(HumanMessage(user_prompt))
        bot_response = start_chatbot(retriever, user_prompt, st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(bot_response["ai"]))

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message,HumanMessage) else "assistant"):
        st.markdown(message)
