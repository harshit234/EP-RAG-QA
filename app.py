import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from loader import load_and_chunk_pdf
from embeddings_store import create_embeddings_and_store, load_existing_store
from chain import create_rag_chain, answer_question

load_dotenv()

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploaded_files'
ALLOWED_EXTENSIONS = {'pdf'}
FAISS_INDEX_NAME = 'faiss_index'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global variable to track if a PDF is loaded
pdf_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Handle PDF upload and create embeddings
    """
    global pdf_loaded
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Only PDF files are allowed'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        print(f"✓ File saved: {temp_path}")
        
        # Create embeddings and vector store
        print("Creating embeddings and vector store...")
        create_embeddings_and_store(temp_path, store_name=FAISS_INDEX_NAME)
        
        print("✓ Vector store created successfully")
        pdf_loaded = True
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'success': True, 
            'message': 'PDF processed successfully!'
        }), 200
    
    except Exception as e:
        print(f"❌ Error in upload: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error processing PDF: {str(e)}'
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Handle question and return answer
    """
    global pdf_loaded
    
    try:
        # Check if PDF is loaded
        if not pdf_loaded or not os.path.exists(FAISS_INDEX_NAME):
            return jsonify({
                'success': False,
                'message': 'Please upload a PDF first'
            }), 400
        
        # Get question from request
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'message': 'Please enter a question'
            }), 400
        
        print(f"🤔 Question: {question}")
        
        # Create RAG chain and get answer
        rag_chain = create_rag_chain(store_name=FAISS_INDEX_NAME)
        answer, source_docs = answer_question(question, rag_chain)
        
        print(f"💬 Answer: {answer}")
        
        # Extract source text
        sources = [doc.page_content for doc in source_docs]
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': sources
        }), 200
    
    except Exception as e:
        print(f"❌ Error in ask: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

@app.route('/debug-env', methods=['GET'])
def debug_env():
    """Debug endpoint to check if API key is loaded"""
    key = os.getenv("GOOGLE_API_KEY")
    return jsonify({
        'GOOGLE_API_KEY_found': key is not None,
        'GOOGLE_API_KEY_preview': f"{key[:10]}...{key[-4:]}" if key else "NOT SET",
        'all_env_keys': [k for k in os.environ.keys() if 'GOOGLE' in k or 'GEMINI' in k or 'API' in k]
    }), 200

if __name__ == '__main__':
    # Run Flask app
    print("[*] Starting RAG Document Q&A Bot...")
    print("[*] Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
