import sys
import os
import logging
import re
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
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from routes.predefined_responses import predefined_responses

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download('punkt')

chat_bp = Blueprint('chatgpt_bp', __name__)

load_dotenv()

dataset_path = './dataset.txt'
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"{dataset_path} does not exist.")

loader = TextLoader(dataset_path, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

index_name = "infinity-demo"

persist_directory = "./chroma_data"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

if not os.access(persist_directory, os.W_OK):
    raise PermissionError(f"The directory {persist_directory} is not writable.")

chroma_client = Client(Settings(persist_directory=persist_directory))
existing_collections = [collection.name for collection in chroma_client.list_collections()]

if index_name not in existing_collections:
    docsearch = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=index_name, persist_directory=persist_directory)
else:
    docsearch = Chroma(collection_name=index_name, persist_directory=persist_directory)

docsearch.persist()

repo_id = "MODEL_ID"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.6, "top_k": 20, "top_p": 0.85, "max_length": 5000},
    huggingfacehub_api_token="HUGGING_FACE_KEY_API"
)

template = """
Act as an expert in providing psychological advice specifically related to autism in children. 
Use the provided context to generate the most accurate and empathetic response regarding autism in children.
Your response should be limited to 2 sentences or 1000 words and summarize the main points of the context.
The input has been translated from Vietnamese to English. 
Provide your response in English, which will be translated to Vietnamese later.
IMPORTANT: Please summarize the main points of the context.

Context: {context}
Question: {question}
Answer: 
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

class ChatBot:
    def __init__(self):
        load_dotenv()
        self.model_name = "MODEL_ID"
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        self.predefined_responses = dict(sorted(predefined_responses.items(), key=lambda x: len(x[0]), reverse=True))

    def compute_angle(self, vec1, vec2):
        cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        return cos_sim

    def extract_keywords(self, text):
        vectorizer = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=5)
        vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()
    
    def translate(self, text, source_lang, target_lang):
        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    def check_predefined_responses(self, user_input):
        lower_input = user_input.lower()
        for keyword, response in self.predefined_responses.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_input):
                return response
        return None
    
bot = ChatBot()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        predefined_response = bot.check_predefined_responses(user_input)
        if predefined_response:
            response = bot.translate(predefined_response, "en_XX", "vi_VN")
            return jsonify({'response': response})
        user_input = bot.translate(user_input, "vi_VN", "en_XX")
        question_vector = embeddings.embed_query(user_input)
        search_results = docsearch.similarity_search(user_input, k=1)
        doc_vectors = [result.page_content for result in search_results]

        for doc_vector in doc_vectors:
            doc_vector_embedding = embeddings.embed_query(doc_vector)
            cos_sim = bot.compute_angle(question_vector, doc_vector_embedding)
            logger.info(f"Vector deflection cos_sim: {cos_sim:.2f}")
            
            if cos_sim >= 0.9:
                keywords = bot.extract_keywords(user_input)
                search_query = " ".join(keywords)
                try:
                    summary = wikipedia.summary(search_query)
                except wikipedia.exceptions.DisambiguationError as e:
                    summary = wikipedia.summary(e.options[0])
                except wikipedia.exceptions.PageError:
                    summary = None

                if summary:
                    result = bot.rag_chain.invoke(f"{summary} {user_input}")
                    answer_start = "Answer: "
                    response = result.split(answer_start)[-1].strip()
                    response = bot.translate(response, "en_XX", "vi_VN")
                    return jsonify({'response': response})

        result = bot.rag_chain.invoke(user_input)   
        answer_start = "Answer: "
        response = result.split(answer_start)[-1].strip()
        response = bot.translate(response, "en_XX", "vi_VN")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
        
