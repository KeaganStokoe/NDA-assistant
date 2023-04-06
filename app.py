from flask import Flask, render_template, request, redirect, url_for
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import json

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

def ingest_pdf():
    loader = PyPDFLoader("agreements/test3.pdf")
    pages = loader.load_and_split()
    return pages

def summarize_agreement(docs):
    prompt_template = """Write a concise summary of the Non-Disclosure agreement below. Use your knowledge as an expert on NDAs to identify the most important aspects of the agreement and include them in your summary. You can use the context below to help you understand the agreement:

    {text}

    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = (
        "Your job is to produce a summary of this Non-Disclosure Agreement. It should provide as much detail to the reader as possible, while remaining concise.\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary and present it in a way that identifies the most salient points. Your aim is to identify the most important aspects of the agreement so that the user does not have to read it themselves.\n"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary. Remember to identify and include the most important aspects of the agreement.\n"
        "If the context isn't useful, return the original summary."
    )

    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    chain = load_summarize_chain(
        ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo'),
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=PROMPT, refine_prompt=refine_prompt
    )

    result = chain({"input_documents": docs}, return_only_outputs=True)

    return result

def identify_red_flags(docs):
    pass

def identify_suggested_improvements(docs):
    pass

@app.route('/success')
def success():
    docs = ingest_pdf()
    agreement_summary = summarize_agreement(docs)
    pretty_result = json.dumps(agreement_summary, indent=4)
    print(pretty_result)
    summary = agreement_summary["output_text"]

    red_flags = identify_red_flags(docs)
    suggested_improvements = identify_suggested_improvements(docs)

    return render_template('success.html', summary=summary, red_flags=red_flags, suggested_improvements=suggested_improvements)

if __name__ == '__main__':
    app.run(debug=True)
