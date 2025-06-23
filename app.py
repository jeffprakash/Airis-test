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

# Custom CSS for styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #262730;
        color: #fff;
        border: 1px solid #464854;
        padding: 15px 15px;
        width: 100%;
        margin: 5px 0px;
    }
    .stButton > button:hover {
        border-color: #6c757d;
        color: #fff;
    }
    .stButton > button:active {
        border-color: #00acb5;
        color: #00acb5;
    }
    .delete-button {
        color: #ff4b4b;
        cursor: pointer;
        float: right;
    }
    .knowledge-item {
        padding: 10px;
        border: 1px solid #464854;
        border-radius: 5px;
        margin: 5px 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title and Configuration ---
st.set_page_config(page_title="Aris: Your Knowledge Hub", layout="wide")
st.title("Aris")
st.caption("A platform that ingests data from PDFs and links and allows you to talk to the knowledge hub.")

# Function to get unique sources from Pinecone
def get_knowledge_items_from_pinecone(index):
    try:
        # Query with empty vector to get all items, limit 10000 to get all
        results = index.query(
            vector=[0] * 4096,  # Empty vector query
            top_k=10000,
            include_metadata=True
        )
        
        # Create a set to store unique sources
        unique_sources = set()
        knowledge_items = []
        
        # Extract unique sources from results
        for match in results['matches']:
            source = match['metadata'].get('source')
            if source and source not in unique_sources:
                unique_sources.add(source)
                # Determine if it's a PDF or URL Hubd on the source
                item_type = "pdf" if source.lower().endswith('.pdf') else "url"
                knowledge_items.append({
                    "type": item_type,
                    "name": source,
                    "source": source
                })
        
        return knowledge_items
    except Exception as e:
        st.error(f"Error fetching knowledge items: {e}")
        return []

# --- Initialization ---
if 'pinecone_index' not in st.session_state:
    with st.spinner("Connecting to Knowledge Hub..."):
        st.session_state.pinecone_index = setup_pinecone_index()
        # Initialize knowledge items from Pinecone
        st.session_state.knowledge_items = get_knowledge_items_from_pinecone(st.session_state.pinecone_index)

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Get started by adding a document or website to the knowledge hub."}]

if 'page' not in st.session_state:
    st.session_state.page = "💬 Chat"

# Function to delete specific knowledge item
def delete_knowledge_item(index):
    item = st.session_state.knowledge_items[index]
    try:
        # Delete from Pinecone Hubd on source
        pinecone_index = st.session_state.pinecone_index
        # Delete vectors with matching metadata source
        pinecone_index.delete(filter={"source": item["source"]})
        # Remove from session state
        st.session_state.knowledge_items.pop(index)
        # Refresh knowledge items from Pinecone
        st.session_state.knowledge_items = get_knowledge_items_from_pinecone(pinecone_index)
        return True
    except Exception as e:
        st.error(f"Error deleting item: {e}")
        return False

# --- Sidebar for navigation and settings ---
with st.sidebar:
    st.header("Navigation")
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("💬 Chat", use_container_width=True, key="nav_chat"):
            st.session_state.page = "💬 Chat"
            st.rerun()
    with col2:
        if st.button("🗂️ Hub", use_container_width=True, key="nav_kb"):
            st.session_state.page = "📚 Knowledge Hub"
            st.rerun()
            
    st.info("Add documents or websites in the 'Knowledge Hub' section, then ask questions in the 'Chat' section.")

# --- Page Content ---
if st.session_state.page == "📚 Knowledge Hub":
    st.header("Manage Your Knowledge Hub")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Add PDF Document")
        pdf_file = st.file_uploader("Upload a PDF document", type="pdf")
        
    with col2:
        st.subheader("🔗 Add Website")
        website_url = st.text_input("Enter a website URL")
    
    # Ingest Button - centered
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Add to Knowledge Hub", use_container_width=True):
            index = st.session_state.pinecone_index
            added = False
            
            if pdf_file:
                try:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress, status):
                        progress_bar.progress(progress / 100)
                        status_text.text(status)

                    with st.spinner(f"Processing {pdf_file.name}..."):
                        upload_and_process_pdf(pdf_file, index, progress_callback=update_progress)
                        st.session_state.knowledge_items = get_knowledge_items_from_pinecone(index)
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✅ Successfully added '{pdf_file.name}' to the knowledge hub.")
                    added = True
                except Exception as e:
                    st.error(f"❌ Failed to process {pdf_file.name}. Error: {e}")

            if website_url:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress, status):
                    progress_bar.progress(progress / 100)
                    status_text.text(status)

                with st.spinner(f"Scraping and processing {website_url}..."):
                    markdown_content = scrape_website(website_url, progress_callback=update_progress)
                    if markdown_content:
                        ingest_data(markdown_content, website_url, index, progress_callback=update_progress)
                        st.session_state.knowledge_items = get_knowledge_items_from_pinecone(index)
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ Successfully added '{website_url}' to the knowledge hub.")
                        added = True
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Failed to scrape or process {website_url}. Please check the URL and try again.")
                        
            if not pdf_file and not website_url:
                st.warning("Please upload a PDF or enter a URL to add to the knowledge hub.")
            
            # Only rerun after all processing is complete
            if added:
                st.rerun()
    
    st.divider()

    # Display Knowledge Items
    st.subheader("Current Knowledge Hub Contents")
    if st.session_state.knowledge_items:
        for idx, item in enumerate(st.session_state.knowledge_items):
            col1, col2 = st.columns([6,1])
            with col1:
                if item["type"] == "pdf":
                    st.markdown(f"""<div class="knowledge-item">📄 <b>PDF:</b> {item['name']}</div>""", unsafe_allow_html=True)
                elif item["type"] == "url":
                    st.markdown(f"""<div class="knowledge-item">🔗 <b>URL:</b> <a href="{item['source']}" target="_blank">{item['name']}</a></div>""", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{idx}"):
                    if delete_knowledge_item(idx):
                        st.success(f"Successfully deleted {item['name']}")
                        st.rerun()
    else:
        st.info("No documents or URLs have been added yet.")

    st.divider()

    # Danger Zone - moved to the bottom
    with st.expander("⚠️ Danger Zone"):
        st.markdown("This will permanently delete all content from your knowledge hub. This action cannot be undone.")
        if st.button("Clear Entire Knowledge Hub", type="primary"):
            try:
                with st.spinner("Deleting all vectors from the Pinecone index..."):
                    index = st.session_state.pinecone_index
                    index.delete(delete_all=True)
                st.session_state.knowledge_items = []
                st.success("✅ Knowledge hub has been cleared.")
                st.rerun()
            except NotFoundException:
                st.info("ℹ️ The knowledge hub is already empty.")
            except Exception as e:
                st.error(f"An error occurred while clearing the knowledge hub: {e}")

# --- Chat Page ---
elif st.session_state.page == "💬 Chat":
    st.header("Chat with your Knowledge Hub")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge hub..."):
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