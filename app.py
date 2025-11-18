import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
import os

# Page config for enhanced UI
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for calming theme
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #d4edda;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Gemini API Key (replace with your key from ai.google.dev)
GEMINI_API_KEY = "AIzaSyDoHEic9INNCDbuhSUcriRJES077oSWgEU" # Get free from https://aistudio.google.com/app/apikey
genai.configure(api_key=GEMINI_API_KEY)

# Function to query Gemini API (robust version)
def query_gemini_llm(prompt, context=""):
    # Trim context if too long (prevent MAX_TOKENS)
    if len(context) > 10000:
        context = context[:10000] + "\n[Context truncated for brevity.]"
    full_prompt = f"{prompt}\n\nContext: {context}" if context else prompt
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    safety_settings = [
        {"category": cat, "threshold": "BLOCK_NONE"}
        for cat in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
    ]
    
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=500, temperature=0.7),
            safety_settings=safety_settings
        )
        
        # Debug: Print token usage (remove in production)
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count if response.usage_metadata else 'N/A'}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count if response.usage_metadata else 'N/A'}")
        print(f"Finish reason: {response.candidates[0].finish_reason if response.candidates else 'None'}")
        
        # Safe text extraction
        if (response.candidates and 
            response.candidates[0].content and 
            response.candidates[0].content.parts):
            return response.candidates[0].content.parts[0].text
        else:
            return "I'm sorry, but I couldn't generate a response right now. This might be due to prompt length or safety guidelines. Try rephrasing or seek professional help for mental health concerns."
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"Error generating response: {str(e)}. Please try again."

# Load and process knowledge base (assume vector_store exists from ingestion)
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vector_store/faiss_index", embeddings, allow_dangerous_deserialization=True)

# Custom prompt for mental health context
prompt_template = """
You are a supportive mental health assistant. Use the following context to provide helpful, empathetic steps to overcome issues like anxiety or depression using therapy techniques (e.g., CBT, mindfulness). Always remind to seek professional help. Do not provide medical advice.

Context: {context}

User: {question}

Response:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Sidebar for resources and mood check
with st.sidebar:
    st.header("🧠 Quick Resources")
    st.markdown("**Helplines:**")
    st.markdown("- **Pakistan:** Rozan Helpline: 0311-7786264 (9 AM – 9 PM, Mon-Sat)")
    st.markdown("- **Pakistan:** Punjab Commission on the Status of Women: 1043 (24/7)")
    st.markdown("- **Pakistan:** Free Mental Health Support (HEC): 0800-69457")
    st.markdown("- **Global:** Befrienders Worldwide: befrienders.org")
    
    st.markdown("**Quick Tips:**")
    st.markdown("- Practice deep breathing: Inhale 4s, hold 4s, exhale 4s")
    st.markdown("- Journal your thoughts daily")
    st.markdown("- Connect with a trusted friend")
    
    st.markdown("---")
    st.header("🌟 How Are You Feeling Today?")
    mood = st.selectbox("Select your mood:", ["😊 Great", "😐 Okay", "😔 A bit low", "😢 Struggling"])
    if st.button("Share More", key="mood_submit"):
        st.session_state.mood_note = mood
        st.rerun()

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 Mental Health Support Chatbot")
with col2:
    st.image("https://img.icons8.com/emoji/48/000000/brain.png", width=50)  # Emoji-like icon

st.markdown("**Welcome!** I'm here to offer empathetic support and evidence-based tips on anxiety, depression, and mental wellness. Remember, I'm not a substitute for professional care—reach out to a therapist if needed. 💙")

# Disclaimer expander
with st.expander("Important Disclaimer", expanded=False):
    st.markdown("""
    This chatbot is for **educational and supportive purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult qualified mental health professionals. If you're in crisis, contact emergency services immediately (e.g., 1043 in Pakistan).
    """)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input(" Share what's on your mind..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # RAG: Retrieve and generate
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking and retrieving helpful info..."):
            docs = retriever.invoke(prompt)
            context = "\n".join([doc.page_content for doc in docs])
            full_prompt = PROMPT.format(context=context, question=prompt)
            response = query_gemini_llm(full_prompt, context)
            st.markdown(response)
        
        # Add feedback buttons
        col_feedback1, col_feedback2 = st.columns(2)
        with col_feedback1:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                st.session_state.feedback = "Helpful"
        with col_feedback2:
            if st.button("👎 Not Helpful", key=f"not_{len(st.session_state.messages)}"):
                st.session_state.feedback = "Not Helpful"
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Powered by Jabiru Labs | For resources, check the sidebar.*")