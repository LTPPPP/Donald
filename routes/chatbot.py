import sys
import os
import logging
from flask import Blueprint, request, jsonify
import nltk
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import Client
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import wikipedia
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from translate import Translator as TranslateTranslator

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
    model_kwargs={"temperature": 0.6, "top_k": 20, "top_p": 0.85, "max_length": 2048},
    huggingfacehub_api_token="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"
)

# Templates
template = """
Bạn là chuyên gia tư vấn tâm lý chuyên về tự kỷ ở trẻ em. Sử dụng ngữ cảnh cung cấp để tạo ra câu trả lời chính xác và gần gũi nhất liên quan đến tự kỷ ở trẻ em.
Câu trả lời của bạn phải giới hạn trong 3 câu hoàn chỉnh và tóm tắt các ý chính của ngữ cảnh.
Câu trả lời của bạn luôn kết thúc bằng dấu chấm ('.').
Câu trả lời của bạn phải bằng tiếng Việt.

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
        self.translator = TranslateTranslator(to_lang="en")
        self.translator_vi = TranslateTranslator(to_lang="vi")

    def compute_angle(self, vec1, vec2):
        cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        return cos_sim

    def extract_keywords(self, text):
        vectorizer = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=5)
        vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()

    def translate_to_english(self, text):
        return self.translator.translate(text)

    def translate_to_vietnamese(self, text):
        return self.translator_vi.translate(text)

bot = ChatBot()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        
        # Detect language and translate if necessary
        user_input = bot.translate_to_english(user_input)
        
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
                try:
                    summary = wikipedia.summary(search_query)
                except wikipedia.exceptions.DisambiguationError as e:
                    summary = wikipedia.summary(e.options[0])
                except wikipedia.exceptions.PageError:
                    summary = None

                if summary:
                    # Run the RAG chain again with the Wikipedia summary
                    result = bot.rag_chain.invoke(f"{summary} {user_input}")
                    print(result)
                    answer_start = "Answer: "
                    response = result.split(answer_start)[-1].strip()
                    response = bot.translate_to_vietnamese(response)
                    return jsonify({'response': response})

        result = bot.rag_chain.invoke(user_input)   
        answer_start = "Answer: "
        response = result.split(answer_start)[-1].strip()
        response = bot.translate_to_vietnamese(response)
         # Append the conversation to the dataset file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            existing_responses = f.read()
        if response not in existing_responses:
            with open(dataset_path, 'a', encoding='utf-8') as f:
                f.write(f"Answer: {response}\n")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
