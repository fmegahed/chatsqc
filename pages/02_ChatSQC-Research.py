import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# to have a custom prompt with langchain
# Ref: https://github.com/hwchase17/langchain/discussions/4199#discussioncomment-5840037
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# LangChain Imports
from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# importing our CSS/HTML templates for decorating the question and response boxes
# the html_templates.py file contains this information 
from html_templates import css, bot_template, user_template


# to change streamlit's default footer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

# to get the page headers
import requests
from bs4 import BeautifulSoup


# to estimate the costs
import tiktoken


# Directory to store embeddings
EMBEDDINGS_DIRECTORY = './vstore/'


# estimate cost of the query and answer
prompt = """
You are a Q&A bot, an intelligent system that answers user questions ONLY based on the information provided by the user. If the information cannot be found in the user information, please say 'As a SQC chatbot grounded only in open-access SQC research papers, I do not know the answer to this question as it is not in my referenced/grounding material. I am sorry for not being able to help.' No answers should be made based on your in-house knowledge. For example, you may know what a large language model is, but that information does not come from the knowledge base that we provided to you. So defining a large language model based on your knowledge is unacceptable. Obviously, other algorithms, descriptions, and formulas that are not in the knowledge base we provided are also unacceptable. The context is:\n{context}.
"""

def estimate_cost(model_name, input_text, output_text):
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Calculate tokens for input and output
    total_input_tokens = len(enc.encode(input_text + prompt))
    output_tokens = len(enc.encode(output_text))

    # Define costs per model
    costs = {
        "gpt-4-1106-preview": {"input": 0.01/1000, "output": 0.03/1000},
        "gpt-3.5-turbo-16k": {"input": 0.0015/1000, "output": 0.002/1000}
    }

    # Estimate the cost
    cost = (total_input_tokens * costs[model_name]["input"]) + (output_tokens * costs[model_name]["output"])

    return cost



# langchain relevant background:
# https://python.langchain.com/docs/ecosystem/integrations/vectara/vectara_chat#pass-in-chat-history
def get_conversation_chain(vectorstore):
    
    # prompting based on:
    # chatClimate and https://github.com/hwchase17/langchain/discussions/4199#discussioncomment-5840037
    system_message_prompt = SystemMessagePromptTemplate.from_template(
      """You are a Q&A bot, an intelligent system that answers user questions ONLY based on the information provided by the user. If the information cannot be found in the user information, please say 'As a SQC chatbot grounded only in open-access SQC research papers, I do not know the answer to this question as it is not in my referenced/grounding material. I am sorry for not being able to help.' No answers should be made based on your in-house knowledge. For example, you may know what a large language model is, but that information does not come from the knowledge base that we provided to you. So defining a large language model based on your knowledge is unacceptable. Obviously, other algorithms, descriptions, and formulas that are not in the knowledge base we provided are also unacceptable. The context is:\n{context}."""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
    )
    
    llm = ChatOpenAI(
        temperature=0.25, model=st.session_state.model_choice
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, input_key = 'question', output_key = 'answer'
    )
        
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(lambda_val=0.025, k=5, filter=None),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={
          "prompt": ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt,
        ]),
    },
    )
    return conversation_chain


# to extract the actual url from the APA citation
def extract_info_from_apa(apa_citation):
    # Extracting URL
    url_start = apa_citation.find('Available at: ') + len('Available at: ')
    url_end = len(apa_citation) -1
    url = apa_citation[url_start:url_end]

    return url



def generate_html_links(search, response_content):
    # Initialize an empty dictionary to store the results
    results = {}

    # If the response content does not include the specified text, process the search data
    if "As a SQC chatbot grounded only in open-access SQC research papers" not in response_content:
        # Iterate through each tuple in the list

        for i, item in enumerate(search):
            # Extract the source
            source = item[0].metadata['source']
            
            page_number = item[0].metadata['page']
            
            score = item[1]
            
            text_chunk = item[0].page_content  # Extract the text chunk

            # If the source is already in our results, append the new text chunk and score
            if source in results:
                results[source].append((score, page_number, text_chunk))
            # If the source is not in our results, add it along with its score and text chunk
            else:
                results[source] = [(score, page_number, text_chunk)]

        # Generate the HTML for the sources and the text chunks
        html_sources = '<div style="padding: 1px;">'
        html_sources += '<i><u>Sources and their relevant text passages:</u></i><br/>'
        for i, (source, chunks) in enumerate(results.items(), start=1):
            apa_citation = st.session_state.headers[source]
            paper_link = extract_info_from_apa(apa_citation)
            html_sources += f'(Source {i}) <a style="font-size:1em;" href="{paper_link}">{apa_citation}</a><ul style="margin-left: 10px; list-style-type:none;">'
            # Sort the chunks by score and enumerate them
            for j, (score, page_number, text_chunk) in enumerate(sorted(chunks), start=1):
                score_rounded = round(float(score), 3)  # round the score to 3 decimal places
                
                # nicely format and print the text chunks upon call
                text_chunk_paragraphs = text_chunk.split('\n\n')  # Split the text chunk into separate paragraphs
                formatted_paragraphs = [f'<p style="font-size: 0.80em;">{paragraph}</p>' for paragraph in text_chunk_paragraphs]  # Format each paragraph with the desired style
                formatted_text_chunk = ' '.join(formatted_paragraphs)  # Join the formatted paragraphs back together
                html_sources += f'<li><details style="font-size: 0.9em;"><summary>Click for relevant text chunk {j} from page {page_number} of the above paper\'s PDF (L2-dist = {score_rounded})</summary><p>{formatted_text_chunk}</p></details></li>'
            
            html_sources += '</ul>'
        html_sources += '</div>'
    else:
        # If the response content includes the specified text, there are no sources or text chunks
        html_sources = ""

    # Return the HTML for the sources
    return html_sources



# a function to handle the user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    # Append new messages to chat history
    st.session_state.chat_history.extend(response['chat_history'])
    
    # things to be added to session state
    model_used = st.session_state.model_choice
    cost = estimate_cost(st.session_state.model_choice, user_question, response['answer'])
    cumulative_cost = sum(item['cost'] for item in st.session_state.queries_data) + cost

    query_data = {
        "index": len(st.session_state.queries_data) + 1,
        "model": model_used,
        "cost": cost,
        "cumulative_cost": cumulative_cost
    }
    
    st.session_state.queries_data.append(query_data)


    # Determine the number of question-response pairs
    num_pairs = len(st.session_state.chat_history) // 2
    
    # Loop over the pairs in reverse order
    for i in range(num_pairs - 1, -1, -1):
        # Get the question and response
        question = st.session_state.chat_history[2*i]
        response = st.session_state.chat_history[2*i + 1]
    
        # Print the question
        st.write(user_template.replace("{{MSG}}", question.content), unsafe_allow_html=True)
        st.write('\n')
        
        # Get the most relevant sources for that question
        search = st.session_state.vectorstore.similarity_search_with_score(question.content, k=5)
        sources_html = generate_html_links(search, response.content)
        
        # Get model, date, and cost info
        last_query = st.session_state.queries_data[i]
        model_date_info = ("<span style='font-size: 0.9em;'>This response was generated on " + 
            datetime.now().strftime('%B %d, %Y') + 
            " using " + last_query['model'] + 
            ". Based on OpenAI's tiktoken library, the estimated cost of this query and answer is " + 
            f"{last_query['cost']:.4f}" + 
            " dollars. The users do not incur these costs, and they are provided solely to provide researchers and practitioners with insights into the expected costs of utilizing the OpenAI APIs to ground ChatGPT to high-quality SQC references.<br/>So far, in this session, you have asked " + 
            str(last_query['index']) + 
            (" question" if last_query['index'] == 1 else " questions") + 
            ", and the cumulative cost of the questions and generated responses within this session is " + 
            f"{last_query['cumulative_cost']:.4f}" + 
            " dollars.</span>")

        st.write(bot_template.replace("{{MSG}}", f"{response.content}<br/><br/>{sources_html}<i><u>Generation date and estimated costs:</u></i><br/>{model_date_info}"), unsafe_allow_html=True)
        #st.write(bot_template.replace("{{MSG}}", f"{response.content}<br/><br/><i><u>Generation date and estimated costs:</u></i><br/>{model_date_info}"), unsafe_allow_html=True)
                


# layout to fix the footer (adapted from https://discuss.streamlit.io/t/st-footer/6447)
def layout(*args):
  style = """
  <style>
      MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      .stApp { bottom: 60px; }
  </style>
  """

  style_div = styles(
      position="fixed",
      right = 0,
      bottom=0,
      margin=px(0, 15, 0, 0),
      text_align="right",
      opacity=0.5,
  )

  body = p()
  foot = div(
      style=style_div
  )(
      body
  )

  st.markdown(style, unsafe_allow_html=True)
  for arg in args:
      if isinstance(arg, str):
          body(arg)
      elif isinstance(arg, HtmlElement):
          body(arg)
  st.markdown(str(foot), unsafe_allow_html=True)


# the footer function
def footer():
    myargs = [
        'Developed with ❤ , but as an open-source tool, it is provided "as is", without warranties or guarantees of any kind. Please verify the generated information independently.'
    ]
    layout(*myargs)



# execute when app is run
def main():
    load_dotenv()
    st.set_page_config(page_title="ChatSQCR", page_icon=":computer:")
    st.write(css, unsafe_allow_html=True)
    
    # overwrite the footer
    footer()

    # Load the preprocessed vectorstore from a local file
    embeddings_model = OpenAIEmbeddings(model = 'text-embedding-ada-002', chunk_size = 1000)
    vectorstore = FAISS.load_local('vstore/vectorstore_papers', embeddings = embeddings_model)
    
    # Load the headers_dictionary
    with open('apa_citations.pkl', 'rb') as f:
        headers_dict = pickle.load(f)
                
    # Variables stored in session state
    st.session_state.vectorstore = vectorstore
    st.session_state.headers = headers_dict
    
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "gpt-3.5-turbo-16k"  # Default model
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if "queries_data" not in st.session_state:
        st.session_state.queries_data = []

    # saving the conversation in session state
    st.session_state.conversation = get_conversation_chain(vectorstore)

   
    # --------------------------------------------------------------------------
    # side bar
    with st.sidebar:
        st.subheader("About ChatSQCR!!")
        
        # Custom css for font size of drop down menu
        st.markdown("""
            <style>
                div[data-baseweb="select"] > div {
                    font-size: 0.85em;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Create a dropdown in the sidebar to let users select a model
        model_mapping = {
            "gpt-3.5-turbo-16k (Model for general queries)": "gpt-3.5-turbo-16k",
            "gpt-4 (State-of-the-art OpenAI model)": "gpt-4-1106-preview",
        }
        
        selected_display_name = st.selectbox(
            "Choose the LLM Model:",
            options=list(model_mapping.keys()),
            index=0 if st.session_state.model_choice == "gpt-3.5-turbo-16k" else 1
        )
    
        # Update the session state with the user's choice
        actual_model_name = model_mapping[selected_display_name]
        st.session_state.model_choice = actual_model_name
        
        
        st.markdown("""
            - **Created by:**
                + :link: [Fadel M. Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)  
                + :link: [Ying-Ju (Tessa) Chen](https://udayton.edu/directory/artssciences/mathematics/chen-ying-ju.php)  
                + :link: [Inez Zwetsloot](https://www.uva.nl/en/profile/z/w/i.m.zwetsloot/i.m.zwetsloot.html)  
                + :link: [Sven Knoth](https://www.hsu-hh.de/compstat/en/sven-knoth-2)  
                + :link: [Douglas C. Montgomery](https://search.asu.edu/profile/10123)  
                + :link: [Allison Jones-Farmer](https://miamioh.edu/fsb/directory/?up=/directory/farmerl2)
        """)
        
        st.write("")
        
        st.markdown("""
            - **Version:** 1.2.0 (Jan 22, 2024)
        
            - **Notes:**
                + This application is built with [Streamlit](https://streamlit.io/) and uses [langchain](https://python.langchain.com/) with OpenAI to provide basic industrial statistics and SQC answers based on all JQT and QE open-access papers (available on Taylor & Francis's Websites by Jan 22, 2024).
                """)

       
    
    # --------------------------------------------------------------------------
    
    # the right-side of the app
    
    st.header("Q&A Open-Access Papers from JQT, Technometrics, and QE")
    st.markdown('**ChatSQC-Research: GPT responses, grounded in open-access SQC research papers.**')
    st.text('\n')
    
    user_question = st.text_input("Ask me a SQC question:", help = "Please input your SQC-related question in the prompt area and press Enter to receive a response. Note that I am specifically designed to answer SQC-related inquiries. For transparency, at the expandable white box below your prompt, I will provide references to relevant sections from the source book that inform my responses.")

    if user_question:
        handle_userinput(user_question)

    # --------------------------------------------------------------------------

if __name__ == '__main__':
    main()


