from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import pprint

def summarize_agreement(nda_path: str, max_pages: int = 3) -> str:
    """
    Given the path to a PDF file containing an NDA agreement, returns a summary of the agreement text and any 
    potential threats or red flags in the agreement using OpenAI's summarization API.
    
    Args:
    - nda_path: str - the file path to the NDA PDF file
    - max_pages: int - the maximum number of pages to summarize (default: 3)
    
    Returns:
    - str - the summarized text of the NDA agreement and any potential threats or red flags
    
    Raises:
    - ValueError: if the provided file path is not a PDF file
    - FileNotFoundError: if the provided file path does not exist
    - Exception: if the summarization API fails to generate a summary
    
    """
    if not nda_path.endswith('.pdf'):
        raise ValueError("File must be a PDF")
    try:
        with open(nda_path, 'rb') as f:
            nda_bytes = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("File not found")
        
    llm = OpenAI(temperature=0)
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(nda_bytes.decode('utf-8'))
    docs = [Document(page_content=t) for t in texts[:max_pages]]
    
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
    chain = MapReduceChain(llm, PROMPT, refine_prompt)
    result = chain({"input_documents": docs}, return_only_outputs=True)
    
    if result and result[0]:
        return result[0].strip()
    else:
        raise Exception("Failed to generate summary")
