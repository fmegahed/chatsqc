EMBEDDINGS_DIRECTORY = './vstore' # directory to store embeddings


# Creating a Citation Dictionary for all our papers:
# --------------------------------------------------
import os
import requests
import urllib.parse
import pickle
import csv
import re
import pandas as pd

from dotenv import load_dotenv


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
# for pdf in pdf_files:
#     print(pdf)



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


def extract_license(license_url):
    match = re.search(r'creativecommons\.org/licenses/([^/]+)/', license_url)
    if match:
        license_code = match.group(1)
        return license_code.replace('-', '').replace('4.0', '-4.0')
    return ""


def format_apa_citation(metadata):
    if 'message' in metadata and 'items' in metadata['message']:
        
        item = metadata['message']['items'][0]  # Assuming the first item is the correct one
        
        authors = item.get('author', [])
        author_str = ', '.join([f"{author['family']}, {author['given']}" for author in authors])
        
        title = item.get('title', [''])[0]
        year = item.get('published-print', {}).get('date-parts', [[None]])[0][0]
        journal = item.get('container-title', [''])[0]
        url = item.get('URL', '')
        license_url = item.get('license', [{}])[0].get('URL', '')  # Get the license URL
        
        apa_citation = f"{author_str} ({year}). {title}. {journal}." + (f" Available at: {url}." if url else "")
        
        # Append license information if available
        if license_url:
            apa_citation += f" License: {license_url}."
        
        return apa_citation
    return "Citation not found"


# -------------------------------------------------------------

# Querying Crossref for metadata and saving APA citations:
# --------------------------------------------------------

# Dictionaries to hold the APA citations
apa_citations = {} # used for pickle file
citation_info = [] # used for csv file

for file_path in pdf_files:
    title = extract_title(file_path)
    metadata = query_crossref(title)
    # step for the pickle file
    if metadata:
        apa_citation = format_apa_citation(metadata)
        apa_citations[file_path] = apa_citation
    # step for the csv file 
    # (could have combined them but got lasy and did not want to change the code above)
    if metadata and 'message' in metadata and 'items' in metadata['message']:
        item = metadata['message']['items'][0]  # Assuming the first item is the correct one
        
        authors = item.get('author', [])
        author_str = ', '.join([f"{author['given']} {author['family']}" for author in authors])
        
        year = item.get('published-print', {}).get('date-parts', [[None]])[0][0]
        title = item.get('title', [''])[0]
        journal = item.get('container-title', [''])[0]
        url = item.get('URL', '')
        license_url = item.get('license', [{}])[0].get('URL', '')
        license = extract_license(license_url)
        citation_dict = {
            'file_path': file_path,
            'authors': author_str,
            'year': year,
            'title': title,
            'journal': journal,
            'url': url,
            'license_url': license_url,
            'license': license
        }
        citation_info.append(citation_dict)

# Saving the dictionary of citations to a pickle file
apa_citations_file = os.path.join('./', 'apa_citations.pkl')
with open(apa_citations_file, 'wb') as f:
  pickle.dump(apa_citations, f)

# Saving the dictionary of citation info to a csv file
csv_filename = 'open_source_refs.csv'
citation_info_df = pd.DataFrame(citation_info)
citation_info_df.to_csv(csv_filename, index=False, encoding='UTF-8')

print(f"CSV file '{csv_filename}' has been created.")



# -------------------------------------------------------------

# Reading and Loading all PDFs:
# -----------------------------
load_dotenv()

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
