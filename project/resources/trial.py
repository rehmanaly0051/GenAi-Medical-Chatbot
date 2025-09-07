from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.schema import Document
from typing import List
import os

load_dotenv()

# Setting the API keys for the Pinecone and OpenAI
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Setting the enviornmenst variables of pinecone and openai
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-chunks'

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
print(index)


# Loading the Document in my case i have the Medical_book
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    document = loader.load()
    return document

extraced_data = load_pdf_files('../data')

# filtering the docs with geeniring it with just the minimal documents inside the document loader
def filtering_minimal_doc(docs: List[Document])->List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {
                    'source': src,
                    'page': doc.metadata.get('page'),
                    'chunk': doc.metadata.get('chunk')
                }
            )
        )
    return minimal_docs

minimal_docs = filtering_minimal_doc(extraced_data)    

# Splitting the document in the chunks for the embeddings
def splitting_text(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )                   
    text = text_splitter.split_documents(minimal_docs)
    return text

text = splitting_text(minimal_docs)

# Downlaoding the embeddings model for the vector embedding of the documents.
def downlaod_embeddings():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
        #model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    return embeddings

embeddings = downlaod_embeddings()
vector_embeddings = embeddings.embed_query('Hello how are you?')
print(vector_embeddings, '\n', len(vector_embeddings))



#print(extraced_data) 
#print(minimal_docs)     
#print(text)
#print(embeddings)