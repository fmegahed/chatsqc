EMBEDDINGS_DIRECTORY = './vstore' # directory to store embeddings


# Creating a Citation Dictionary for all our papers:
# --------------------------------------------------
import os
import requests
import urllib.parse
import pickle

# Define the base directory
base_dir = 'papers\\pdfs\\'

# Initialize an empty list to store the paths of PDF files
pdf_files = []

# Walk through the directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.pdf'):
            # Construct the full path of the file
            full_path = os.path.join(root, file)

            # Add the full path to the list
            pdf_files.append(full_path)

# Printing all found PDF file paths
for pdf in pdf_files:
    print(pdf)



def extract_title(path):
    return os.path.basename(path).rsplit('.', 1)[0]

def query_crossref(title):
    query = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.title={query}&rows=1"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def format_apa_citation(metadata):
    if 'message' in metadata and 'items' in metadata['message']:
        
        item = metadata['message']['items'][0]  # Assuming the first item is the correct one
        
        authors = item.get('author', [])
        author_str = ', '.join([f"{author['family']}, {author['given']}" for author in authors])
        
        title = item.get('title', [''])[0]
        year = item.get('published-print', {}).get('date-parts', [[None]])[0][0]
        journal = item.get('container-title', [''])[0]
        url = item.get('URL', '')
        
        apa_citation = f"{author_str} ({year}). {title}. {journal}." + (f" Available at: {url}." if url else "")
        
        return apa_citation
    return "Citation not found"


# Dictionary to hold the APA citations
apa_citations = {}

for file_path in pdf_files:
    title = extract_title(file_path)
    metadata = query_crossref(title)
    if metadata:
        apa_citation = format_apa_citation(metadata)
        apa_citations[file_path] = apa_citation

# Saving the dictionary of citations
apa_citations_file = os.path.join('./', 'apa_citations.pkl')
with open(apa_citations_file, 'wb') as f:
  pickle.dump(apa_citations, f)





# Reading and Loading all PDFs:
# -----------------------------

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFDirectoryLoader("./papers/pdfs/")

our_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
  )

data = loader.load_and_split(text_splitter=our_text_splitter)

print(data)



# Creating the Vectorstore:
# -------------------------
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI's state of the art embedding model (surprisingly also their cheapest)
embeddings_model = OpenAIEmbeddings(model = 'text-embedding-ada-002', chunk_size = 1000)

# get embeddings for the data and create the vectorstore
vectorstore = FAISS.from_documents(documents = data, embedding=embeddings_model)

vectorstore.save_local(os.path.join(EMBEDDINGS_DIRECTORY, f'vectorstore_papers'))
