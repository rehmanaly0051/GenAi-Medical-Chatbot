from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
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

# Setting the environment variables of pinecone and openai
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

# Loading the Document in my case i have the Medical_book
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    document = loader.load()
    return document

extracted_data = load_pdf_files('../data')

# Splitting the document in the chunks for the embeddings FIRST
def splitting_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )                   
    text = text_splitter.split_documents(docs)
    return text

# Split first, then filter with chunk information
split_docs = splitting_text(extracted_data)

# Filtering the docs with generating it with just the minimal documents inside the document loader
def filtering_minimal_doc(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source', '')
        page = doc.metadata.get('page')
        
        # Create clean metadata with only valid values
        metadata = {
            'source': str(src) if src else '',
            'chunk_id': str(i)  # Add chunk ID based on position
        }
        
        # Only add page if it's not None
        if page is not None:
            metadata['page'] = int(page) if isinstance(page, (int, float, str)) else 0
            
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )
    return minimal_docs

minimal_docs = filtering_minimal_doc(split_docs)

# Downloading the embeddings model for the vector embedding of the documents.
def download_embeddings():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
        #model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    return embeddings

embeddings = download_embeddings()
vector_embeddings = embeddings.embed_query('Hello how are you?')
print(vector_embeddings, '\n', len(vector_embeddings))

# Storing the data inside the Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=minimal_docs,
    embedding=embeddings,
    index_name=index_name
)

print("Documents successfully stored in Pinecone!")
print(docsearch)