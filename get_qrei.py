import os
import time
import random
import pandas as pd


# A Function to Get the URLs and Put them in a CSV so we can manually download them
# Wiley blocked our automated attempts to download the PDFs, 
# so we had to manually download them
# ------------------------------------------------------------------------------

# Define a function to return the url
def get_url(doi):
    return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true"

# Load the CSV file into a pandas DataFrame with 'ISO-8859-1' encoding
df = pd.read_csv('data/qrei_open_access_list_wiley.csv', encoding='ISO-8859-1')

# Filter the DataFrame based on the "Licence" column
filtered_df = df[df['Licence '].isin(['CC BY', 'CC BY-NC'])]

# Extract the DOIs from the filtered DataFrame
dois = filtered_df['DOI']

# Loop through the DOIs and download the PDFs
qrei_urls = []
for doi in dois:
    qrei_urls.append(get_url(doi))

# Save the URLs to a CSV file
pd.DataFrame(qrei_urls).to_csv('data/pdf_urls.csv', index=False, header='URL')


# ------------------------------------------------------------------------------

# Rewriting the pdf file names

# Directory path where the PDF files are located
directory_path = 'papers/pdfs/qrei/'

import os

# Directory path where the PDF files are located
directory_path = 'papers/pdfs/qrei/'

# Iterate over the files in the directory
for filename in os.listdir(directory_path):
    # Split the filename into parts using ' - '
    parts = filename.split(' - ')
    
    # Extract the last part (title)
    title = parts[-1].strip()
    
    # Create the new filename
    new_filename = f"{title}"
    
    # Rename the file
    os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))
    print(f"{new_filename}")
