import sys
import os
import logging
from flask import Blueprint, request, jsonify, Flask
import nltk
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb import Client
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from googletrans import Translator
from langdetect import detect
import wikipedia
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download('punkt')

chat_bp = Blueprint('chatgpt_bp', __name__)

# Initialize environment variables
load_dotenv()

# Ensure dataset file exists
dataset_path = './dataset.txt'
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"{dataset_path} does not exist.")

# Update the TextLoader to specify the encoding
loader = TextLoader(dataset_path, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)
print(docs)

embeddings = HuggingFaceEmbeddings()

index_name = "infinity-demo"

# Ensure the persist directory exists and has write permissions
persist_directory = "./chroma_data"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Check if the directory has write permissions
if not os.access(persist_directory, os.W_OK):
    raise PermissionError(f"The directory {persist_directory} is not writable.")

# Initialize Chroma client
chroma_client = Client(Settings(persist_directory=persist_directory))

# Check if the collection already exists
existing_collections = [collection.name for collection in chroma_client.list_collections()]

if index_name not in existing_collections:
    # Create new collection and add documents
    docsearch = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=index_name, persist_directory=persist_directory)
else:
    # Load the existing collection
    docsearch = Chroma(collection_name=index_name, persist_directory=persist_directory)

# Persist the index to disk
docsearch.persist()

# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 1, "top_k": 64, "top_p": 0.95, "max_length": 1000},
    huggingfacehub_api_token="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"
)

# Templates
template = """
You are an expert psychological consultant specializing in autism in children. Use the context provided to generate the most accurate and closest answers related to autism in children. 
Your response must be limited to a maximum of 3 complete sentences and should summarize the main ideas of the context.
Your response always end with a period ('.').

Context: {context}
Question: {question}
Answer: 
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Initialize the chatbot
class ChatBot:
    def __init__(self):
        load_dotenv()
        self.rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def compute_angle(self, vec1, vec2):
        cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        return cos_sim

    def extract_keywords(self, text):
        vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, max_features=5)
        vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()

bot = ChatBot()
translator = Translator()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        # Retrieve document and question vectors
        question_vector = embeddings.embed_query(user_input)
        search_results = docsearch.similarity_search(user_input, k=1)
        doc_vectors = [result.page_content for result in search_results]

        # Calculate and log angles
        for doc_vector in doc_vectors:
            doc_vector_embedding = embeddings.embed_query(doc_vector)
            cos_sim = bot.compute_angle(question_vector, doc_vector_embedding)
            logger.info(f"Vector deflection cos_sim: {cos_sim:.2f}")
            
            if cos_sim >= 0.6:
                keywords = bot.extract_keywords(user_input)
                search_query = " ".join(keywords)
                wikipedia.set_lang('vi')
                try:
                    summary = wikipedia.summary(search_query)
                except wikipedia.exceptions.DisambiguationError as e:
                    summary = wikipedia.summary(e.options[0])
                except wikipedia.exceptions.PageError:
                    summary = None

                if summary:
                    # Run the RAG chain again with the Wikipedia summary
                    result = bot.rag_chain.invoke(f"{summary} {user_input}")
                    answer_start = "Answer: "
                    response = result.split(answer_start)[-1].strip()
                    translated_response = translator.translate(response, dest='vi').text
                    return jsonify({'response': translated_response})

        result = bot.rag_chain.invoke(user_input)   
        answer_start = "Answer: "
        response = result.split(answer_start)[-1].strip()
        translated_response = translator.translate(response, dest='vi').text
        return jsonify({'response': translated_response})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
