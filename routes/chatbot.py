# routes/chatbot.py

from flask import Blueprint, request, jsonify
import google.generativeai as genai
import nltk
from rake_nltk import Rake
import os
import markdown
from pylatexenc.latex2text import LatexNodes2Text
import PIL.Image
import html
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from .API import API_KEY_GEMINI, MODEL_NAME, VISION_MODEL, SPREADSHEET_ID, RANGE_NAME

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Rake
r = Rake()

# Set up Gemini API
genai.configure(api_key=API_KEY_GEMINI)
model = genai.GenerativeModel(MODEL_NAME)
vision_model = genai.GenerativeModel(VISION_MODEL)

# Create Blueprint
chat_bp = Blueprint('chat', __name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

<<<<<<< HEAD
# Initialize the main template
MAIN_TEMPLATE = """
Chủ nhân, người tạo ra bạn là Lâm Tấn Phát là chàng trai đẹp trai, tài năng và rất thích học hỏi.
Bạn là trợ lý giáo dục chuyên về tư vấn vấn đề tâm lý ở trẻ tự kỷ. Hãy:
1. Cung cấp giải thích chính xác và chi tiết.
2. Trả lời bằng tiếng Việt.
3. Có thể phân tích hình ảnh và trả lời câu hỏi từ hình ảnh.
4. Nhận diện vật thể từ hình ảnh và cung cấp thông tin chi tiết.
5. Tóm tắt câu trả lời lại tối đa 1000 chữ.
6. Chỉ trả lời tóm tắt đúng ý.
Hãy trả lời ngắn gọn nhưng đầy đủ. Nếu cần thêm thông tin, hãy hỏi người dùng.
=======
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
>>>>>>> 30f09b65bf7225c5b7e7050feb9e04de007f7854
"""

# Dictionary to store user context
user_context = {}

<<<<<<< HEAD
def generate_response(prompt):
    response = model.generate_content(prompt).text
    return response
=======
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
>>>>>>> 30f09b65bf7225c5b7e7050feb9e04de007f7854

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_conversation(user_id, user_input, response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "user_input": user_input,
        "response": response
    }

    log_directory = "../documents/response"
    ensure_directory_exists(log_directory)

    json_log_path = os.path.join(log_directory, "log.json")
    try:
        with open(json_log_path, "r+", encoding="utf-8") as json_file:
            try:
                logs = json.load(json_file)
            except json.JSONDecodeError:
                logs = []
            logs.append(log_entry)
            json_file.seek(0)
            json.dump(logs, json_file, indent=2)
    except FileNotFoundError:
        with open(json_log_path, "w", encoding="utf-8") as json_file:
            json.dump([log_entry], json_file, indent=2)

    txt_log_path = os.path.join(log_directory, "log.txt")
    with open(txt_log_path, "a", encoding="utf-8") as txt_file:
        txt_file.write(f"[{timestamp}] User {user_id}:\n")
        txt_file.write(f"Input: {user_input}\n")
        txt_file.write(f"Response: {response}\n\n")

@chat_bp.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        img = PIL.Image.open(filepath)
        prompt = f"{MAIN_TEMPLATE}\nDescribe this image in detail"
        response = vision_model.generate_content([prompt, img])
        # Delete the file after processing
        os.remove(filepath)
        
        return jsonify({'response': markdown.markdown(LatexNodes2Text().latex_to_text(response.text))})

@chat_bp.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    user_id = request.remote_addr
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        user_context.pop(user_id, None)
        response = "Bái bai! Hẹn gặp lại bạn sau nhé! moah moah <3"
        log_conversation(user_id, user_input, response)
        return jsonify({'response': response})

    global MAIN_TEMPLATE
    if user_input.lower().startswith('adjust template:'):
        MAIN_TEMPLATE = user_input[16:].strip()
        response = "Template adjusted successfully."
        log_conversation(user_id, user_input, response)
        return jsonify({'response': response})

    if user_id not in user_context:
        user_context[user_id] = []

    user_context[user_id].append(f"User: {user_input}")

    context = "\n".join(user_context[user_id])
    prompt = f"{MAIN_TEMPLATE}\n\n{context}\n\nChatbot:"    
    try:
        response = generate_response(prompt)
        response = html.escape(response)
        response = markdown.markdown(response)
        print("response: " + response)
        user_context[user_id].append(f"Chatbot: {response}")
        log_conversation(user_id, user_input, response)
        return jsonify({'response': response})
    except Exception as e:
<<<<<<< HEAD
        error_message = "An error occurred while processing your request. Please try again."
        log_conversation(user_id, user_input, error_message)
        return jsonify({'response': error_message})
=======
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
        
>>>>>>> 30f09b65bf7225c5b7e7050feb9e04de007f7854
