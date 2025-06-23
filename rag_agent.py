import os
import PyPDF2
import requests
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import firebase_admin
from firebase_admin import credentials, storage
import io
import time

# --- Configuration ---
load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = "https://uae-demo-2-vhze9sb.svc.aped-4627-b74a.pinecone.io"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")

# Build Firebase credentials from environment variables
FIREBASE_CREDENTIALS_DICT = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN"),
}

if not all([
    NEBIUS_API_KEY, PINECONE_API_KEY, GEMINI_API_KEY, PINECONE_HOST, FIRECRAWL_API_KEY, FIREBASE_STORAGE_BUCKET,
    FIREBASE_CREDENTIALS_DICT["type"], FIREBASE_CREDENTIALS_DICT["project_id"], FIREBASE_CREDENTIALS_DICT["private_key_id"],
    FIREBASE_CREDENTIALS_DICT["private_key"], FIREBASE_CREDENTIALS_DICT["client_email"], FIREBASE_CREDENTIALS_DICT["client_id"],
    FIREBASE_CREDENTIALS_DICT["auth_uri"], FIREBASE_CREDENTIALS_DICT["token_uri"], FIREBASE_CREDENTIALS_DICT["auth_provider_x509_cert_url"],
    FIREBASE_CREDENTIALS_DICT["client_x509_cert_url"], FIREBASE_CREDENTIALS_DICT["universe_domain"]
]):
    raise ValueError("API keys, Pinecone host, or Firebase credentials are missing. Please check your .env file and script configuration.")

# --- Clients ---
# Nebius LLM client for embeddings
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
)

# Pinecone client
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "uae-demo-2"
EMBEDDING_DIMENSION = 4096  # Make sure this matches your embedding model's output dimension.

# Firecrawl client
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# Firebase Client
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_DICT)
    firebase_admin.initialize_app(cred, {
        'storageBucket': FIREBASE_STORAGE_BUCKET
    })
    print("✅ Firebase App initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Firebase App: {e}")
    # Depending on the use case, you might want to raise the exception
    # raise e


import re

def clean_markdown_content(markdown_list):
    if isinstance(markdown_list, str):
        markdown_list = [markdown_list]

    cleaned_text = []
    for md in markdown_list:
        # Remove markdown links but keep the display text: [text](url) → text
        md = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
        
        # Remove image markdown entirely: ![](url)
        md = re.sub(r'!\[.*?\]\([^\)]+\)', '', md)

        # Remove citation-style links like [\[24\]]
        md = re.sub(r'\[\s*\\?\[\d+\\?\]\s*\]', '', md)
        
        # Remove markdown headings (##, ###, etc.)
        md = re.sub(r'#+\s?', '', md)

        # Remove table formatting and stray pipes
        md = re.sub(r'\|', '', md)

        # Remove unicode bullets or special formatting characters
        md = re.sub(r'[•·\uf0b7]', '', md)

        # Remove excessive whitespace and newlines
        md = re.sub(r'\n+', '\n', md)
        md = re.sub(r'\s{2,}', ' ', md)

        # Strip leading/trailing spaces
        md = md.strip()
        
        cleaned_text.append(md)

    return "\n\n".join(cleaned_text)


# --- PDF and Text Processing ---
def upload_and_process_pdf(pdf_file, index, progress_callback=None):
    """Uploads PDF to Firebase, extracts text, and ingests it."""
    file_name = pdf_file.name
    
    if progress_callback:
        progress_callback(0, "Starting PDF upload...")
    
    # 1. Upload to Firebase Storage
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"pdfs/{file_name}")
        
        pdf_file.seek(0)
        blob.upload_from_file(pdf_file, content_type='application/pdf')
        print(f"☁️ Uploaded '{file_name}' to Firebase Storage at gs://{bucket.name}/{blob.name}")
        
        if progress_callback:
            progress_callback(20, "PDF uploaded to storage, extracting text...")

        # 2. Extract text from the PDF file in memory
        print(f"📄 Reading text from {file_name}...")
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages)
        print(f"✅ Extracted {len(text)} characters.")
        
        if progress_callback:
            progress_callback(40, "Text extracted, processing chunks...")
        
        # 3. Ingest the extracted text data
        ingest_data(text, file_name, index, progress_callback)
        
    except Exception as e:
        print(f"❌ An error occurred during PDF processing: {e}")
        raise


def scrape_website(url, progress_callback=None):
    """Scrapes a website and returns its content in markdown format."""
    if progress_callback:
        progress_callback(0, "Starting website scraping...")
    
    print(f"🕸️  Scraping {url} with Firecrawl...")
    scrape_result = firecrawl_app.scrape_url(url,formats=['markdown', 'html'])
    
    if progress_callback:
        progress_callback(50, "Website scraped, cleaning content...")

    cleaned_markdown = clean_markdown_content(scrape_result.markdown)
    
    if progress_callback:
        progress_callback(70, "Content cleaned...")
    
    return cleaned_markdown



def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    print(f"📦 Created {len(chunks)} text chunks.")
    return chunks

# --- Embeddings ---
def get_embedding(text):
    """Generates embedding for a text using Nebius."""
    response = nebius_client.embeddings.create(
        model="BAAI/bge-en-icl",
        input=text
    )
    return response.data[0].embedding

# --- Pinecone Vector DB ---
def setup_pinecone_index():
    """Connects to a specific Pinecone index using the provided host."""
    print(f"🌲 Connecting to Pinecone index '{INDEX_NAME}' at host {PINECONE_HOST}...")
    index = pinecone.Index(name=INDEX_NAME, host=PINECONE_HOST)
    print("✅ Connection successful.")
    return index

def ingest_data(text_content, source_name, index, progress_callback=None):
    """Chunks text, creates embeddings, and upserts them into Pinecone."""
    if progress_callback:
        progress_callback(40, "Creating text chunks...")
    
    chunks = chunk_text(text_content)
    total_chunks = len(chunks)
    
    print(f"🧠 Generating embeddings and preparing vectors for {source_name}...")
    vectors = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            # Calculate progress from 50 to 90 based on chunk processing
            chunk_progress = 50 + int((i / total_chunks) * 40)
            progress_callback(chunk_progress, f"Processing chunk {i+1}/{total_chunks}...")
        
        embedding = get_embedding(chunk)
        vector = {
            "id": f"{source_name}-chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk, "source": source_name}
        }
        vectors.append(vector)
        
    if not vectors:
        print(f"No vectors created for {source_name}. Skipping ingestion.")
        return

    if progress_callback:
        progress_callback(90, "Uploading to knowledge hub...")

    print(f"⬆️  Upserting {len(vectors)} vectors to Pinecone...")
    # Upsert in batches to avoid overwhelming the API
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    
    if progress_callback:
        progress_callback(100, "Complete!")
    
    print(f"✅ Ingestion complete for {source_name}.")

# --- RAG Pipeline ---
def call_gemini_api(prompt):
    """Calls the Gemini API using a direct REST request."""
    print("🤖 Asking Gemini for an answer...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    body = {
      "contents": [{
        "parts": [{"text": prompt}]
      }]
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response found.')
        return text
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Gemini API: {e}")
        return "Sorry, I encountered an error while trying to generate an answer."

def query_agent(question, index, k=30):
    """Queries the agent to get an answer for a given question."""
    print(f"\n❓ User Question: {question}")
    
    # 1. Create embedding for the question
    question_embedding = get_embedding(question)
    
    # 2. Query Pinecone for similar contexts
    print("🔍 Searching for relevant context in Pinecone...")
    results = index.query(
        vector=question_embedding,
        top_k=k,
        include_metadata=True
    )
    contexts = [match['metadata']['text'] for match in results['matches']]

    print("contexts",contexts)



    # 3. Build the prompt for Gemini
    prompt = f"""
    You are a helpful assistant. Use the following context from a document to answer the question.
    If the context doesn't contain the answer, state that you couldn't find the information in the document.
    NOTE: DO NOT USE ANYTHING OTHER THAN THE CONTEXT TO ANSWER THE QUESTION.

    Context:
    ---
    {" ".join(contexts)}
    ---

    Question: {question}

    Answer:
    """
    
    # 4. Call Gemini to generate the answer
    answer = call_gemini_api(prompt)
    
    return answer

# --- Main Execution ---
# if __name__ == "__main__":
#     # --- Source Configuration ---
#     pdf_file_path = "asdadasun.pdf"
#     website_url = "https://en.wikipedia.org/wiki/Sun"

#     # Setup Pinecone Index
#     pinecone_index = setup_pinecone_index()

#     # --- Ingest PDF Document ---
#     if os.path.exists(pdf_file_path):
#         pdf_text = extract_text_from_pdf(pdf_file_path)
#         ingest_data(pdf_text, os.path.basename(pdf_file_path), pinecone_index)
#     else:
#         print(f"⚠️  Warning: PDF file '{pdf_file_path}' not found. Skipping PDF ingestion.")

#     # --- Ingest Website Content ---
#     if website_url:
#         markdown_content = scrape_website(website_url)
#         if markdown_content:
#             ingest_data(markdown_content, website_url, pinecone_index)


#     # Example question that could be answered by your PDF (adjust question as needed)
#     question_from_pdf = "explain the General characteristics of the sun" # Assuming your PDF has this info
#     answer_2 = query_agent(question_from_pdf, pinecone_index)
    
#     print("\n💡 Final Answer (from PDF):")
#     print(answer_2) 