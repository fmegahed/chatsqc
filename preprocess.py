

# needed libraries
# ----------------------------------------------------------------------------

import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS



# Path for the pdf, pkl (HTML pages) and vstore:
# -----------------------------------------------------------------------------
PRELOADED_PDF_FILE = './pdfs/montgomery6thedition.pdf' # path to pdf file
PKL_DIRECTORY = './ehandbook/data.pkl'

EMBEDDINGS_DIRECTORY = './vstore' # directory to store embeddings

# Functions:
# -----------------------------------------------------------------------------

## a function that reads the HTML documents stored in the pickle file
def get_pickle_text(pickle_file):
    with open(pickle_file, "rb") as file:
        html_docs = pickle.load(file) # using pickle
        
        # now we will put the text in one long string and return it
        text = ""
        for document in  range(len(html_docs)):
            text += html_docs[document].page_content
    
    return text



## a function that reads a pdf using the Py2PDF2 library and returns a single 
## string containing all the text from the PDF
def get_pdf_text(pdf_file):
    with open(pdf_file, "rb") as file:
        read_pdf = PdfReader(file) # using the PyPDF2 library
        
        # getting the meta data of the read pdf
        # not returned by function, but extracted for potential future use
        pdf_meta_data = read_pdf.metadata
        pdf_outline = read_pdf.outline
        
        # now we will put the text in one long string and return it
        text = ""
        for page in read_pdf.pages:
            text += page.extract_text()
    
    return text



## a function that splits the string based on a new line with a max size of 2000
## returns chunks of text, which we will use the embeddings to convert them to
## a numeric list of matrices capturing the relationship between the words
def get_text_chunks(text):
    
    # we used the Recursive CharacterTextSplitter since it is the recommended splitter
    # for generic text per the the langChain documentation
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators= ["\n\n", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # cleaning up the chunks prior to embedding
    # this step follow's OpenAI's example in 
    # https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb
    # chunks = str(chunks).replace("\n", " ") 
    
    return chunks


## a function which applies our embeddings to our text chunks and returns a
## data base (vector store) containing the numeric representation of relationship
## between text
def get_vectorstore(text_chunks):
    
    # OpenAI's state of the art embedding model (surprisingly also their cheapest)
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', chunk_size = 1000)
    
    # FAISS selected based on https://js.langchain.com/docs/modules/indexes/vector_stores/ 
    # (since it does not need any other servers to stand up)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


## the main function which will run once we run the file
def main():
    load_dotenv()

    # Determine the type of text and preprocess accordingly
    text_type = 'HTML'  # Change this to 'PDF' to process a PDF file instead

    if text_type == 'PDF':
        raw_text = get_pdf_text(PRELOADED_PDF_FILE)
        if PRELOADED_PDF_FILE == './pdfs/montgomery6thedition.pdf':
            raw_text = raw_text[34008:1787375]  # remove preface (begining) & appendix (end)
    elif text_type == 'HTML':
        raw_text = get_pickle_text(PKL_DIRECTORY)
    else:
        raise ValueError(f'Invalid text_type: {text_type}')

    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)

    # Store the vectorstore in a local file with a suffix that reflects the type of file
    with open(os.path.join(EMBEDDINGS_DIRECTORY, f'vectorstore_{text_type.lower()}.pkl'), 'wb') as f:
        pickle.dump(vectorstore, f)
    
    print("Done")



if __name__ == '__main__':
    main()


