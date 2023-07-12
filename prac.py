import streamlit as st
import os
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()
os.getenv('OPENAI_API_KEY')

llm = OpenAI(verbose=True)

st.set_page_config(page_title='Play With Your Data', 
                   page_icon='📚'
                   )
st.title("ASK YOUR PDF💬")
pdf = 'PATH_OF_YOUR_PDF'

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size = 1000,
            chunk_overlap = 200,
            # length_function = len
        )

    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

        
    user_query = st.text_input("Enter the question:")

    if user_query:
        docs = knowledge_base.similarity_search(user_query)

        chain = load_qa_chain(llm=llm, chain_type='stuff')
        response = chain.run(input_documents = docs, question = user_query)
        st.write(response)



        


