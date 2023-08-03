import os
import requests
from bs4 import BeautifulSoup
import re
from langchain.document_loaders import SeleniumURLLoader
import pickle

# Retrieving all the Webpages from the NIST E-handobbk:
# ------------------------------------------------------------------------------
# retrieving the XML data
ehandbook = requests.get('https://www.itl.nist.gov/div898/handbook/sitemap.xml')

# parsing the returned XML
soup = BeautifulSoup(ehandbook.content, 'xml')

# extracting all the URLs
urls = [element.text for element in soup.find_all('loc')]


# Focusing only on the URLs associated with one of the 8 chapters of the book
# -----------------------------------------------------------------------------
# Regular expression pattern
pattern = re.compile(r"https://www\.itl\.nist\.gov/div898/handbook/(\w+)/section(\d+)/(\w+)(\d+)\.htm")

# List to store the URLs and their section and subsection numbers
url_data = []

# Define the custom order
order = ['eda', 'mpc', 'ppc', 'pmd', 'pri', 'pmc', 'prc', 'apr']
order_rank = {identifier: rank for rank, identifier in enumerate(order)}

for url in urls:
    match = pattern.search(url)
    if match:
        identifier = match.group(1)
        section = int(match.group(2))
        subsection_identifier = match.group(3)
        subsection = int(match.group(4))
        url_data.append((identifier, section, subsection_identifier, subsection, url))


# Sorting the URLs
# ------------------------------------------------------------------------------

sorted_urls = sorted(url_data, key=lambda x: (order_rank[x[0]], x[1], x[2], x[3]))

# from tuples to list
sorted_urls_only = [url for identifier, section, subsection_identifier, subsection, url in sorted_urls]


# Custom function to scrape the headers
# ------------------------------------------------------------------------------
def scrape_headers(urls):
    headers_dict = {}
    
    # Iterate over each URL
    for url in urls:
        # Fetch the page content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Concatenate the <h2> headers
        headers = ' '.join(header.text for header in soup.find_all('h2'))
        
        # Add the headers to the dictionary, with the URL as the key
        headers_dict[url] = headers

    # Save the headers dictionary as a pickle file
    with open('./ehandbook/headers_dict.pkl', 'wb') as f:
        pickle.dump(headers_dict, f)

    print("Headers saved to headers_dict.pkl")


scrape_headers(sorted_urls_only)
