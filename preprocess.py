"""
This is a script to create a vectorstore for the NIST eHandbook of Engineering Statistics.
"""



# ----- Loading the required libraries and functions ------------------------

import os
import pickle
from dotenv import load_dotenv

# langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Path for the pdf, pkl (HTML pages) and vstore:
# -----------------------------------------------------------------------------
PKL_DIRECTORY = './ehandbook/data.pkl'

EMBEDDINGS_DIRECTORY = './vstore' # directory to store embeddings


load_dotenv()


# custom functions:
# -----------------------------------------------------------------------------

def get_pickle_text(pickle_file):
    with open(pickle_file, "rb") as file:
        html_docs = pickle.load(file) # using pickle
        
        source_text_list = []
        for document in html_docs:
            source = document.metadata['source']
            text = document.page_content
            text_to_remove = '''\n\nSite Privacy\n\nAccessibility\n\nPrivacy Program\n\nCopyrights\n\nVulnerability Disclosure\n\nNo Fear Act Policy\n\nFOIA\n\nEnvironmental Policy\n\nScientific Integrity\n\nInformation Quality Standards\n\nCommerce.gov\n\nScience.gov\n\nUSA.gov\n\nVote.gov'''
            cleaned_text = text.replace(text_to_remove, '')
            source_text_list.append((source, cleaned_text))
    
    return source_text_list



def get_text_chunks(source_text_list):
    # instantiate the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators= ["\n\n", ""]
    )
    
    source_chunk_list = []
    for source, text in source_text_list:
        # split the text into chunks
        chunks = text_splitter.split_text(text)
        
        # associate each chunk with the source
        for chunk in chunks:
            source_chunk_list.append((source, chunk))
    
    return source_chunk_list



def get_vectorstore(source_chunk_list):
    # OpenAI's state of the art embedding model
    embeddings_model = OpenAIEmbeddings(model = 'text-embedding-ada-002', chunk_size = 1000)

    # separate the texts from their sources
    texts = [chunk for source, chunk in source_chunk_list]
    metadata = [{'source': source} for source, chunk in source_chunk_list]

    # get embeddings for the texts and associate with the metadata
    vectorstore = FAISS.from_texts(texts=texts, metadatas=metadata, embedding=embeddings_model)

    return vectorstore





# Using the functions
# ---------------------------------------------------------------------------

def main():
    raw_text = get_pickle_text(PKL_DIRECTORY)
    
    # based on the computation below, the largest chunk had 82,523 characters
    # so we have to chunk it
    max_length = max(len(text) for source, text in raw_text)
    
    text_chunks = get_text_chunks(raw_text)
    
    vectorstore = get_vectorstore(text_chunks)
    
    # Store the vectorstore in a local file with a suffix that reflects the type of file
    vectorstore.save_local(os.path.join(EMBEDDINGS_DIRECTORY, f'vectorstore_nist'))
    
    print("Done")



if __name__ == '__main__':
    main()
