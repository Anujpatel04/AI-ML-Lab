#!/usr/bin/env python3
"""
Flask web application for LinkedIn Roster Agent
"""

import os
import sys
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from linkedin_roster_agent import LinkedInRosterAgent
import tempfile
import shutil

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

DEFAULT_API_KEY = "AIzaSyAHUpx_fWs5cqlHs8DHbQYMtColfoAVRoY"
agent = LinkedInRosterAgent(api_key=os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY))

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_profile():
    """Process LinkedIn profile screenshot and return annotated image."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)'}), 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{filename}")
        file.save(input_path)
        
        output_filename = f"rostered_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        result = agent.process_profile(input_path, output_path)
        
        if output_path.lower().endswith('.png'):
            mimetype = 'image/png'
        elif output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            mimetype = 'image/jpeg'
        else:
            mimetype = 'image/jpeg'
        
        return send_file(
            output_path,
            mimetype=mimetype
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing profile: {error_msg}", file=sys.stderr)
        
        if "quota" in error_msg.lower() or "429" in error_msg or "rate limit" in error_msg.lower():
            return jsonify({
                'error': 'API quota exceeded. The free tier allows 20 requests per day. Please wait or check your API usage at https://ai.dev/usage?tab=rate-limit'
            }), 429
        else:
            return jsonify({'error': error_msg}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_profile():
    """Analyze LinkedIn profile and return JSON data."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)'}), 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{filename}")
        file.save(input_path)
        
        profile_data = agent.analyze_profile(input_path)
        roster_suggestions = agent.generate_roster_suggestions(profile_data)
        
        return jsonify({
            'success': True,
            'profile_data': profile_data,
            'roster_suggestions': roster_suggestions
        })
        
    except Exception as e:
        print(f"Error analyzing profile: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    import atexit
    atexit.register(lambda: shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True))
    
    port = int(os.getenv('PORT', 5001))
    
    print("Starting LinkedIn Roster Agent Web Server...")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
