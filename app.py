import streamlit as st
from rag_agent import (
    setup_pinecone_index,
    scrape_website,
    ingest_data,
    query_agent,
    clean_markdown_content,
    upload_and_process_pdf
)
from pinecone.exceptions import NotFoundException

# --- App Title and Configuration ---
st.set_page_config(page_title="Chat with your Knowledge Base", layout="wide")
st.title("🧠 Chat with your Knowledge Base")
st.caption("Powered by Nebius, Pinecone, and Gemini")

# --- Sidebar for navigation and settings ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["💬 Chat", "📚 Knowledge Base"])
    st.info("Add documents or websites in the 'Knowledge Base' section, then ask questions in the 'Chat' section.")


# --- Initialization ---
# Initialize Pinecone index in session state
if 'pinecone_index' not in st.session_state:
    with st.spinner("Connecting to Knowledge Base..."):
        st.session_state.pinecone_index = setup_pinecone_index()

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Get started by adding a document or website to the knowledge base."}]


# --- Page Content ---

# --- Knowledge Base Page ---
if page == "📚 Knowledge Base":
    st.header("Manage Your Knowledge Base")
    st.markdown("Add documents (PDFs) or website URLs to build your knowledge base. The content will be processed, embedded, and stored for the chat agent to use.")
    
    # PDF Uploader
    pdf_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Website URL Input
    website_url = st.text_input("Or add a website URL")
    
    # Ingest Button
    if st.button("Add to Knowledge Base"):
        index = st.session_state.pinecone_index
        
        # Process PDF file
        if pdf_file:
            try:
                with st.spinner(f"Processing {pdf_file.name}..."):
                    upload_and_process_pdf(pdf_file, index)
                st.success(f"✅ Successfully added '{pdf_file.name}' to the knowledge base.")
            except Exception as e:
                st.error(f"❌ Failed to process {pdf_file.name}. Error: {e}")

        # Process Website URL
        if website_url:
            with st.spinner(f"Scraping and processing {website_url}..."):
                markdown_content = scrape_website(website_url)
                if markdown_content:
                    ingest_data(markdown_content, website_url, index)
                    st.success(f"✅ Successfully added '{website_url}' to the knowledge base.")
                else:
                    st.error(f"❌ Failed to scrape or process {website_url}. Please check the URL and try again.")
        
        if not pdf_file and not website_url:
            st.warning("Please upload a PDF or enter a URL to add to the knowledge base.")
    
    st.divider()

    st.header("⚠️ Danger Zone")
    st.markdown("This will permanently delete all content from your knowledge base. This action cannot be undone.")
    if st.button("Clear Knowledge Base", type="primary"):
        try:
            with st.spinner("Deleting all vectors from the Pinecone index..."):
                index = st.session_state.pinecone_index
                index.delete(delete_all=True)
            st.success("✅ Knowledge base has been cleared.")
        except NotFoundException:
            st.info("ℹ️ The knowledge base is already empty.")
        except Exception as e:
            st.error(f"An error occurred while clearing the knowledge base: {e}")


# --- Chat Page ---
elif page == "💬 Chat":
    st.header("Chat with your documents and websites")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge base..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                index = st.session_state.pinecone_index
                response = query_agent(prompt, index)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 