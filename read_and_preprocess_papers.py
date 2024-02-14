"""
This is a script to read and preprocess the papers in the 'papers/pdfs' directory. 
It served several purposes:
  - Create a citation dictionary for all our papers saved as a pickle file
  (we use this dictionary to provide in-text citations for the generated answers).  
  - Create a CSV file with the citation information for all our papers
  (we put this on GitHub so readers can easily see the papers used in our ChatSQC-Research)
  - Removing the Cover Pages of all Technometrics and QE papers.
  - Creating a vectorstore for all our papers, which we read using the `DirectoryLoader` 
  and `PyMuPDFLoader` classes from the `langchain` package. The vectorstore is then saved
  as a local file in the 'vstore' directory.
  
Things to Consider in the Future:
  - We could also consider removing the headers and footers from the papers . 
  (this would require a more complex approach to the text extraction)
  - We could also consider combining shorter blocks with the preceding text.
  - We could also remove the references from the papers.
"""


EMBEDDINGS_DIRECTORY = './vstore' # directory to store embeddings


# Creating the needed functions for our citation dictionary and CSV file:
# ----------------------------------------------------------------------
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

# Delete first page of QE and Technometrics papers:
# -------------------------------------------------
import fitz
import os

# Function to check and delete the first page if it contains specific texts
def delete_first_page_if_conditions_met(file_path):
    # Open the PDF file
    doc = fitz.open(file_path)
    
    # Check if the document has at least one page
    if len(doc) > 0:
        # Extract text from the first page
        first_page_text = doc[0].get_text()
        
        # Conditions to check in the first page
        conditions = ["Submit your article to this journal", "To cite this article:"]
        
        # Check if both conditions are met
        if all(condition in first_page_text for condition in conditions):
            # Select all pages except the first one
            pages_to_keep = list(range(1, len(doc)))
            doc.select(pages_to_keep)
            
            # Overwrite the original PDF
            doc.save(file_path,  incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    
    # Close the document
    doc.close()

# Directories to search for PDFs
directories = ['./papers/pdfs/qe/', './papers/pdfs/tech/']

# Loop through each directory
for directory in directories:
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PDF
        if filename.endswith('.pdf'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Process the file
            delete_first_page_if_conditions_met(file_path)

print("First pages of QE and Technometrics papers have been deleted.")


# -------------------------------------------------------------

# Reading and Loading all PDFs:
# -----------------------------
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader("./papers/pdfs/", loader_cls = PyMuPDFLoader)

our_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
  )

data = loader.load_and_split(text_splitter=our_text_splitter)

def clean_data(documents):
    download_pattern = r"Downloaded from https:\/\/onlinelibrary\.wiley\.com\/doi\/[\w.\/]+ by Miami \[.*?\].*?OA articles are governed by the applicable Creative Commons License"
    line_break_pattern = r"-\n"
    
    for doc in documents:
        # Clean the page_content attribute directly
        cleaned_content = re.sub(download_pattern, "", doc.page_content, flags=re.DOTALL)
        cleaned_content = re.sub(line_break_pattern, "\n", cleaned_content)
        doc.page_content = cleaned_content
    return documents

cleaned_data = clean_data(data)

# Showing the cleaned page_content of the last document in the list as an example
print(cleaned_data[len(cleaned_data)-1].page_content)


# -------------------------------------------------------------

# Creating the Vectorstore:
# -------------------------
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI's state of the art embedding model
embeddings_model = OpenAIEmbeddings(model = 'text-embedding-ada-002', chunk_size = 1000)

# get embeddings for the data and create the vectorstore
vectorstore = FAISS.from_documents(documents = cleaned_data, embedding=embeddings_model)

vectorstore.save_local(os.path.join(EMBEDDINGS_DIRECTORY, f'vectorstore_papers'))
