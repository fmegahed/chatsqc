import streamlit as st

st.set_page_config(page_title="ChatSQC", layout="wide")

def main():
    # Page title and introduction
    st.title("ChatSQC: A Template for Conversational SQC Applications")
    st.write("""
    Welcome to ChatSQC, a versatile chat application offering two distinct modes of interaction:
    
    - **ChatSQCB**: A base version of the ChatSQC application, grounded in the NIST/SEMATECH Handbook of Engineering Statistics.
    - **ChatSQCR**: A research-grade version of the ChatSQC application, grounded in [the entire collection of CC-BY and CC-BY-NC open-access journal articles from: (a) Technometrics, (b) Quality Engineering, and (c) QREI](https://raw.githubusercontent.com/fmegahed/chatsqc/main/open_source_refs.csv).

    Explore **each mode by selecting them from the sidebar**!
    """)

    # Any additional information or instructions
    st.markdown("## How to Use ChatSQC?")
    st.write("""
    - Use the sidebar to navigate between the Base and Research modes.
    - Pick the LLM of your choice from the dropdown menu.
    - Start asking your own questions.
    """)
    
    with st.sidebar:
        st.subheader("About ChatSQC!!")
        
        # Custom css for font size of drop down menu
        st.markdown("""
            <style>
                div[data-baseweb="select"] > div {
                    font-size: 0.85em;
                }
            </style>
        """, unsafe_allow_html=True)
        
        
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
            - **Version:** 1.3.0 (Feb 03, 2024)
        """)

       

# Ensure this script runs only in standalone mode (not necessary for Streamlit pages, but good practice)
if __name__ == "__main__":
    main()
