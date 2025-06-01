import streamlit as st
import requests
import fitz  # PyMuPDF
import time
import json
from typing import Dict, Any, Optional

# Set up Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("📚 RAG Chatbot Interface")
st.markdown("Interact with the RAG chatbot by uploading documents or searching concert info for an artist.")

API_URL = "http://127.0.0.1:8000"

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "document_count" not in st.session_state:
    st.session_state.document_count = 0

def extract_text_from_pdf(uploaded_pdf):
    """Extract text from a PDF file."""
    try:
        # Create a temporary file path
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.getvalue())
        
        # Use PyMuPDF to extract text
        doc = fitz.open("temp.pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"🚫 Error extracting text from PDF: {e}")
        return ""

def make_api_request(endpoint: str, data: Dict[str, Any], is_json: bool = False) -> Optional[Dict]:
    """Make an API request."""
    try:
        if is_json:
            response = requests.post(f"{API_URL}/{endpoint}", json=data, timeout=30)
        else:
            response = requests.post(f"{API_URL}/{endpoint}", data=data, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🚫 Connection Error: Could not connect to the API server. Is it running?")
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout: The request took too long to complete.")
    except requests.exceptions.HTTPError as e:
        error_msg = "Unknown error"
        try:
            error_data = response.json()
            if "detail" in error_data:
                error_msg = error_data["detail"]
            elif "error" in error_data:
                error_msg = error_data["error"]
        except:
            error_msg = str(e)
        st.error(f"❌ HTTP Error: {error_msg}")
    except json.JSONDecodeError:
        st.error("🚫 Invalid response from server (not valid JSON)")
    except Exception as e:
        st.error(f"🚫 Error: {str(e)}")
    return None

# --- Sidebar with document count ---
with st.sidebar:
    st.header("Stats")
    st.metric("Documents Loaded", st.session_state.document_count)
    
    st.markdown("---")
    st.header("Chat History")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")
        
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

# --- Upload or Paste Document Section ---
st.markdown("### 📤 Upload or Paste a Document")

with st.form("upload_form"):
    st.session_state.uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    st.session_state.manual_text = st.text_area("Or paste your document text here:", value=st.session_state.manual_text)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        upload_button = st.form_submit_button("Submit Document")
    with col2:
        if st.session_state.document_count > 0:
            st.write(f"✅ {st.session_state.document_count} documents loaded")

    if upload_button:
        text = st.session_state.manual_text.strip()
        uploaded_file = st.session_state.uploaded_file

        if uploaded_file and text:
            st.warning("⚠️ Please use only one input: either upload a file OR paste text.")
        elif not uploaded_file and not text:
            st.warning("⚠️ Please provide either a file or some pasted text.")
        else:
            with st.spinner("Processing document..."):
                try:
                    if uploaded_file:
                        if uploaded_file.type == "application/pdf":
                            # Extract text from the PDF
                            content = extract_text_from_pdf(uploaded_file)
                            if not content.strip():
                                st.error("🚫 Could not extract text from PDF. Is it a scanned document?")
                                st.stop()
                        else:
                            content = uploaded_file.read().decode("utf-8")
                    else:
                        content = text
                    
                    if len(content.strip()) < 10:
                        st.error("🚫 Document is too short. Please provide more content.")
                        st.stop()
                    
                    data = make_api_request("ingest", {"document": content})
                    
                    if data:
                        message = data.get("message", "")
                        summary = data.get("summary", "")
                        chunks_count = data.get("chunks_count", 0)
                        
                        if "cannot ingest" in message.lower():
                            st.warning(f"⚠️ {message}")
                        else:
                            st.success("✅ Document processed successfully!")
                            st.markdown(f"**Summary:** {summary}")
                            st.session_state.document_count += 1
                        
                        st.session_state.manual_text = ""
                        st.session_state.uploaded_file = None
                        
                        # Add to chat history
                        if "cannot ingest" not in message.lower():
                            st.session_state.chat_history.append(("user", f"[Uploaded a document]"))
                            st.session_state.chat_history.append(("bot", f"Document processed. Summary: {summary}"))
                
                except Exception as e:
                    st.error(f"🚫 Error: {e}")

st.markdown("---")

# --- Ask a Question Section ---
st.markdown("### 💬 Ask a Question Based on Uploaded Documents")

with st.form("question_form"):
    user_input = st.text_input("Type your question here:")
    ask_button = st.form_submit_button("Ask")

    if ask_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            # Add user question to history
            st.session_state.chat_history.append(("user", user_input))
            
            with st.spinner("Thinking..."):
                data = make_api_request("ask", {"query": user_input}, is_json=True)
                
                if data:
                    answer = data.get("answer", "No answer returned.")
                    
                    # Add bot response to history
                    st.session_state.chat_history.append(("bot", answer))
                    
                    # Display the answer
                    st.success("✅ Answer:")
                    st.markdown(f"**{answer}**")

st.markdown("---")

# --- Search Concerts by Artist ---
st.markdown("### 🎤 Search Upcoming Concerts for an Artist or Band")

with st.form("artist_form"):
    artist_name = st.text_input("Enter an artist or band name:")
    search_button = st.form_submit_button("Search Concerts")

    if search_button:
        if artist_name.strip() == "":
            st.warning("⚠️ Please enter an artist or band name.")
        else:
            # Add search to history
            st.session_state.chat_history.append(("user", f"[Searched concerts for: {artist_name}]"))
            
            with st.spinner("Searching..."):
                data = make_api_request("search_concerts", {"query": artist_name}, is_json=True)
                
                if data:
                    concerts = data.get("result", "No results found.")
                    
                    # Add results to history
                    st.session_state.chat_history.append(("bot", f"Concert results for {artist_name}: {concerts}"))
                    
                    # Display results
                    st.success(f"🎶 Upcoming Concerts for {artist_name}:")
                    st.markdown(f"{concerts}")
