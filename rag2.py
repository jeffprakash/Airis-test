import os
import PyPDF2
import requests
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# --- Configuration ---
load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = "https://uae-demo-2-vhze9sb.svc.aped-4627-b74a.pinecone.io"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not all([NEBIUS_API_KEY, PINECONE_API_KEY, GEMINI_API_KEY, PINECONE_HOST, FIRECRAWL_API_KEY]):
    raise ValueError("API keys or Pinecone host are missing. Please check your .env file and script configuration.")

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
def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    print(f"📄 Reading text from {pdf_path}...")
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)
    print(f"✅ Extracted {len(text)} characters.")
    return text

def scrape_website(url):
    """Scrapes a website and returns its content in markdown format."""
    print(f"🕸️  Scraping {url} with Firecrawl...")
    scrape_result = firecrawl_app.scrape_url(url,formats=['markdown', 'html'])

    # print("scrape_result",type(scrape_result),"markdown",scrape_result.markdown)

    cleaned_markdown = clean_markdown_content(scrape_result.markdown)

    # print("cleaned_markdown",cleaned_markdown)
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

def ingest_data(text_content, source_name, index):
    """Chunks text, creates embeddings, and upserts them into Pinecone."""
    chunks = chunk_text(text_content)
    
    print(f"🧠 Generating embeddings and preparing vectors for {source_name}...")
    vectors = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
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

    print(f"⬆️  Upserting {len(vectors)} vectors to Pinecone...")
    # Upsert in batches to avoid overwhelming the API
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
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

def query_agent(question, index, k=25):
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
if __name__ == "__main__":
    # --- Source Configuration ---
    pdf_file_path = "asdadasun.pdf"
    website_url = "https://en.wikipedia.org/wiki/Sun"

    # Setup Pinecone Index
    pinecone_index = setup_pinecone_index()

    # --- Ingest PDF Document ---
    if os.path.exists(pdf_file_path):
        pdf_text = extract_text_from_pdf(pdf_file_path)
        ingest_data(pdf_text, os.path.basename(pdf_file_path), pinecone_index)
    else:
        print(f"⚠️  Warning: PDF file '{pdf_file_path}' not found. Skipping PDF ingestion.")

    # --- Ingest Website Content ---
    if website_url:
        markdown_content = scrape_website(website_url)
        if markdown_content:
            ingest_data(markdown_content, website_url, pinecone_index)


    # Example question that could be answered by your PDF (adjust question as needed)
    question_from_pdf = "explain the General characteristics of the sun" # Assuming your PDF has this info
    answer_2 = query_agent(question_from_pdf, pinecone_index)
    
    print("\n💡 Final Answer (from PDF):")
    print(answer_2) 