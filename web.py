import os
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

llm = OpenAI(verbose=True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    pdf = request.files['pdf']
    if pdf and pdf.filename.endswith('.pdf'):
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = request.form['user_question']
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            return response
    return "Invalid PDF file or question."

if __name__ == '__main__':
    app.run(debug=True)
