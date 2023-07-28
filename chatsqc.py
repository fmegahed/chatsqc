import os
import pickle
import streamlit as st
from dotenv import load_dotenv

# to have a custom prompt with langchain
# Ref: https://github.com/hwchase17/langchain/discussions/4199#discussioncomment-5840037
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# typical langchain imports
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# importing our CSS/HTML templates for decorating the question and response boxes
# the html_templates.py file contains this information 
from html_templates import css, bot_template, user_template


# to change streamlit's default footer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


# Directory to store embeddings
EMBEDDINGS_DIRECTORY = './vstore'


# langchain relevant background:
# https://python.langchain.com/docs/ecosystem/integrations/vectara/vectara_chat#pass-in-chat-history
def get_conversation_chain(vectorstore):
    
    # prompting based on:
    # chatClimate and https://github.com/hwchase17/langchain/discussions/4199#discussioncomment-5840037
    system_message_prompt = SystemMessagePromptTemplate.from_template(
    "You are a Q&A bot, an intelligent system that answers user questions ONLY based on the information provided by the user. When you use the information provided by the user, please include '\n (Source: NIST/SEMATECH e-Handbook of Statistical Methods)' at the end of your response with a line break. If the information cannot be found in the user information, please say 'As a SQC chatbot grounded only in NIST/SEMATCH's Engineering Staistics Handbook, I do not know the answer to this question as it is not in my referenced/grounding material. I am sorry for not being able to help.' No answers should be made based on your in-house knowledge. For example, you may know what a large language model is, but that information does not come from the knowledge base that we provided to you. So defining a large language model based on your knowledge is unacceptable. Obviously, other algorithms, descriptions, and formulas that are not in the knowledge base we provided are also unacceptable. The context is:\n{context}."
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
    )
    
    llm = ChatOpenAI(
        temperature=0.25,
        model="gpt-3.5-turbo-16k" 
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
        
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(lambda_val=0.025, k=5, filter=None),
        memory=memory,
        combine_docs_chain_kwargs={
          "prompt": ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt,
        ]),
    },
    )
    return conversation_chain


# a function to handle the user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    
    # Append new messages to chat history
    st.session_state.chat_history.extend(response['chat_history'])

    # st.write(response) # used for debugging
    
    # commented code below will print the question-answer pairs in the order
    # by which they were asked (similar to ChatGPT)
    # -- can get tedious without the ability to automatically jump to the latest
    # answer which is not implemented in our app
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write('\n')
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    
    # With a streamlit expander
    with st.expander('Click for the two most relevant text chunks based on your last prompt, and then click to hide before the next prompt.'):
        # Find the relevant pages based on:
        # https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss#:~:text=Similarity%20Search%20with%20score%E2%80%8B&text=The%20returned%20distance%20score%20is,a%20lower%20score%20is%20better.&text=It%20is%20also%20possible%20to,parameter%20instead%20of%20a%20string.
        search = st.session_state.vectorstore.similarity_search_with_score(user_question)
        # Write out the top two most relevant snippets
        st.markdown(f'<span style="color:#C3142D;font-weight:bold;"> {"The ChatSQC identifies several chunks of text that are closely associated with the question and then uses them to augment ChatGPTs knowledge base with additional context prior to generating the response. Below, we provide you a sample of the two most relevant text chunks. Note that: (a) we provide this to highlight potential cases where ChatSQC is solely relying on ChatGPT in making the response, and (b) help the user identify scenarios where there does not seem to be a relevant chunk of text."}</span>', unsafe_allow_html=True)
        st.write(search[0][0].page_content)
        st.write('The L2 distance associated with the query and the text above is ', search[0][1], ', where a lower value indicates a better similarity to the reference text.')
        st.write("----------------------------------------------------")
        st.write(search[1][0].page_content)
        st.write('The L2 distance associated with the query and the text above is ', search[1][1], ', where a lower value indicates a better similarity to the reference text.')
    
    # also used for debugging
    # # Print the chat history after each user input
    # st.write("Chat history:")
    # for message in st.session_state.chat_history:
    #     st.write(message.content)

    
    # Determine the number of question-response pairs
    num_pairs = len(st.session_state.chat_history) // 2
    
    # Loop over the pairs in reverse order
    for i in range(num_pairs - 1, -1, -1):
        # Get the question and response
        question = st.session_state.chat_history[2*i]
        response = st.session_state.chat_history[2*i + 1]
    
        # Print the question and response
        st.write(user_template.replace("{{MSG}}", question.content), unsafe_allow_html=True)
        st.write('\n')
        st.write(bot_template.replace("{{MSG}}", response.content), unsafe_allow_html=True)
    



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
    st.set_page_config(page_title="ChatSQC", page_icon=":computer:")
    st.write(css, unsafe_allow_html=True)

    # Load the preprocessed vectorstore from a local file
    with open(os.path.join(EMBEDDINGS_DIRECTORY, 'vectorstore_html.pkl'), 'rb') as f:
        vectorstore = pickle.load(f)
    
    # Store vectorstore in session state
    st.session_state.vectorstore = vectorstore
    
    # Initialize chat_history in session state if it doesn't already exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # saving the conversation in session state
    st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Q&A the NIST/SEMATECH Handbook :books:")
    st.markdown('**ChatSQC: GPT responses, grounded in industrial statistics and quality control theory.**')
    st.text('\n')
    
    user_question = st.text_input("Ask me a SQC question:", help = "Please input your SQC-related question in the prompt area and press Enter to receive a response. Note that I am specifically designed to answer SQC-related inquiries. For transparency, at the expandable white box below your prompt, I will provide references to relevant sections from the source book that inform my responses.")


    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("About ChatSQC!!")
        st.markdown("""
            - **Created by:** 
                + :link: [Fadel M. Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)
                + :link: [Ying-Ju (Tessa) Chen](https://udayton.edu/directory/artssciences/mathematics/chen-ying-ju.php)
                + :link: [Inez Zwetsloot](https://www.uva.nl/en/profile/z/w/i.m.zwetsloot/i.m.zwetsloot.html)
                + :link: [Sven Knoth](https://www.hsu-hh.de/compstat/en/sven-knoth-2)
                + :link: [Douglas C. Montgomery](https://search.asu.edu/profile/10123)
                + :link: [Allison Jones-Farmer](https://miamioh.edu/fsb/directory/?up=/directory/farmerl2)
                    
            - **Version:** 0.2.0
                
            - **Last Updated:** July 24, 2023
            
            - **Notes:**
                + This application is built with [Streamlit](https://streamlit.io/) and uses [langchain](https://python.langchain.com/) with OpenAI to provide basic industrial statistics and SQC answers based on the seminal [NIST/SEMATECH Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/index.htm).
                + To construct ChatSQC, we adapted the excellent [MultiPDF Chat App by @alejandro-ao](https://github.com/alejandro-ao/ask-multiple-pdfs) such that the preprocessing of our reference (HTML) documents are done offline to save time and dollars.
        """)

    footer()

    
if __name__ == '__main__':
    main()


