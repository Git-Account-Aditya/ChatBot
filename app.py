from flask import Flask, request, render_template, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence
from collections import defaultdict
import os
from dotenv import load_dotenv
import markdown
import uuid

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
db = SQLAlchemy(app)

# Database model
class ChatSession(db.Model):
    id = db.Column(db.String, primary_key=True)
    title = db.Column(db.String, default="New Chat")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String, db.ForeignKey('chat_session.id'))
    sender = db.Column(db.String)  # 'user' or 'bot'
    content = db.Column(db.Text)

with app.app_context():
    db.create_all()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192")
prompt_template = PromptTemplate(
    input_variables=["message", "chat_history"],
    template=(
        "You are Chatbot, a highly knowledgeable, friendly, and concise AI assistant. "
        "You remember the ongoing conversation and use it to provide contextually relevant, accurate, and helpful answers. "
        "If the user asks for code, provide clear and well-formatted code blocks. "
        "If the user asks for a list or steps, use bullet points or numbers. "
        "If you don't know the answer, say so honestly. "
        "\n\n"
        "Conversation so far:\n{chat_history}\n"
        "User: {message}\n"
        "Assistant:"
    )
)

# Fix: Ensure output_parser returns a dict with 'output' key and chain returns only the value for ConversationBufferMemory
class OutputParserWithKey(StrOutputParser):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        return {"output": result}

output_parser = OutputParserWithKey()

# Use the new output_parser in the chain
def run_chain(llm_input):
    # The chain returns a dict with 'output' key, but we want just the value for buffer memory
    result = (prompt_template | llm | output_parser).invoke(llm_input)
    # Debug: print result type and value
    print('DEBUG run_chain result:', result, type(result))
    if isinstance(result, dict) and 'output' in result:
        return result['output']
    elif isinstance(result, str):
        # Fallback: treat as string (should not happen if output_parser is used)
        return result
    else:
        raise ValueError(f"Unexpected result from chain: {result}")

# Global dictionary to hold buffer memory per session
session_buffers = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    input_key="message",  # Set input_key to match prompt and chain
    output_key="output",  # Explicitly set output_key
    return_messages=True
))

def get_or_create_session():
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
        new_session = ChatSession(id=session['chat_session_id'])
        db.session.add(new_session)
        db.session.commit()
    return session['chat_session_id']

def get_or_create_buffer(session_id):
    # If buffer exists, return it
    if session_id in session_buffers:
        return session_buffers[session_id]
    # Otherwise, initialize from DB
    buffer = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="message",  # Set input_key to match prompt and chain
        output_key="output",  # Explicitly set output_key
        return_messages=True
    )
    messages = Message.query.filter_by(session_id=session_id).all()
    # Fix: Always pass dicts to save_context, and ensure both keys are present
    for m in messages:
        # Always pass both keys as dicts, and ensure both are strings (not None)
        if m.sender == 'user':
            buffer.save_context({"message": m.content or ""}, {"output": ""})
        else:
            buffer.save_context({"message": ""}, {"output": m.content or ""})
    session_buffers[session_id] = buffer
    return buffer

@app.route('/', methods=['GET'])
def home_page():
    # Get all chat sessions for the sidebar
    sessions = ChatSession.query.all()
    return render_template('index.html', sessions=sessions)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    session_id = get_or_create_session()
    chat_session = None  # Ensure chat_session is always defined
    if not user_message:
        return jsonify({'response': "No message received."}), 400
    try:
        # Get or create buffer for this session
        buffer = get_or_create_buffer(session_id)
        
        # Prepare chat history from buffer
        chat_history = buffer.load_memory_variables({})['chat_history']
        # Compose prompt (add user_info if you want)
        llm_input = {
            'message': user_message,
            'chat_history': chat_history
        }
        response_text = run_chain(llm_input)

        # Update buffer memory
        buffer.save_context({"message": user_message}, {"output": response_text})

        # Save user and bot messages to DB
        db.session.add(Message(session_id=session_id, sender='user', content=user_message))
        db.session.add(Message(session_id=session_id, sender='bot', content=response_text))

        # Update session title if it's still "New Chat"
        chat_session = ChatSession.query.get(session_id)
        if chat_session and (not chat_session.title or chat_session.title == "New Chat"):
            chat_session.title = user_message[:30]  # Use first 30 chars of user message
        db.session.commit()

        # Convert Markdown to HTML for better formatting
        response_html = markdown.markdown(
            response_text, 
            extensions=['fenced_code', 'codehilite']
        )
    except Exception as e:
        print(f"Error: {e}")
        response_html = "Sorry, something went wrong. Please try again."
    # Return session info for sidebar update
    return jsonify({'response': response_html, 'session': {'id': session_id, 'title': chat_session.title if chat_session else "New Chat"}})

@app.route('/sessions', methods=['GET'])
def get_sessions():
    sessions = ChatSession.query.all()
    return jsonify([{'id': s.id, 'title': s.title} for s in sessions])

@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    messages = Message.query.filter_by(session_id=session_id).all()
    history = [
        {'sender': m.sender, 'content': markdown.markdown(m.content, extensions=['fenced_code', 'codehilite'])}
        for m in messages
    ]
    session['chat_session_id'] = session_id  # Switch session
    return jsonify(history)

@app.route('/new_session', methods=['POST'])
def new_session():
    session['chat_session_id'] = str(uuid.uuid4())
    new_session = ChatSession(id=session['chat_session_id'])
    db.session.add(new_session)
    db.session.commit()
    # Return the new session's id and title for sidebar update
    return jsonify({'id': session['chat_session_id'], 'title': new_session.title})

@app.route('/delete_session/<session_id>', methods=['POST'])
def delete_session(session_id):
    Message.query.filter_by(session_id=session_id).delete()
    ChatSession.query.filter_by(id=session_id).delete()
    db.session.commit()
    return jsonify({'success': True})

#------------------------------------------------------------------------#

@app.route('/features')
def features():
    return render_template('features.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)