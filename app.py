from flask import Flask, render_template, request, redirect, url_for
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
text_splitter = CharacterTextSplitter()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('agreements/' + file.filename)
    return redirect(url_for('success'))


def summarize_agreement():
    loader = PyPDFLoader("agreements/test1.pdf")
    pages = loader.load_and_split()

    # docs = [Document(page_content=t) for t in pages]

    prompt_template = """Write a concise summary of the following:

    {text}

    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = (
        "Your job is to produce a final summary that includes any potential threats or red flags in the agreement\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary and present any threats or red flags in the agreement, as well as suggestions for improvement\n"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary, and list any potential threats or red flags in the agreement\n"
        "If the context isn't useful, return the original summary."
    )

    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    chain = load_summarize_chain(
        OpenAI(temperature=0),
        chain_type="map_reduce",
        return_intermediate_steps=True
    )

    result = chain({"input_documents": pages}, return_only_outputs=True)

    return result

@app.route('/success')
def success():
    summary = summarize_agreement()
    red_flags = "red flags"
    suggested_improvements = "improvements"
    return render_template('success.html', summary=summary, red_flags=red_flags, suggested_improvements=suggested_improvements)

if __name__ == '__main__':
    app.run(debug=True)
