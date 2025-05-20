from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
import traceback  # Added for better error logging
import tempfile
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import re
import spacy

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not found!")

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini model initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    # Set a flag to disable Gemini functionality if it fails to initialize
    gemini_available = False
else:
    gemini_available = True

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes
app.config['PROPAGATE_EXCEPTIONS'] = True  # Add this to see detailed errors

chat_session = None
current_field = None
assessment_in_progress = False
question_count = 0
absolute_max_questions = 30  # Safety limit to prevent infinite questioning

# -------------------- Home --------------------
@app.route('/')
def index():
    return "Flask App: Random Forest API & Gemini Career Assessment"

# -------------------- Normalisation function --------------------
def normalize_text(text):
    if not isinstance(text, str):
        return text
    return text.replace("'", "'").replace(""", "\"").replace(""", "\"")

# -------------------- Gemini Assessment Routes --------------------
@app.route('/start_assessment', methods=['POST'])
def start_assessment():
    global chat_session, current_field, assessment_in_progress, question_count
    
    try:
        # Check if Gemini is available
        if not gemini_available:
            return jsonify({"error": "Gemini API is not available. Check server logs."}), 500
            
        # Get data from request
        data = request.get_json()
        if not data or 'field' not in data:
            return jsonify({"error": "Field is required"}), 400
            
        # Reset conversation state
        question_count = 0
        assessment_in_progress = True
        
        # Get the field and additional info from the request
        current_field = data['field']
        education_level = data.get('education_level', 'Not specified')
        prior_exposure = data.get('prior_exposure', 'Not specified')
        
        # Initialize chat session with empty history
        chat_session = gemini_model.start_chat(history=[])
        
        # Send system prompt with the additional user information
        initial_prompt = f"""
        You are an assessment assistant for the field of {current_field}. Your task is to assess if the user is suitable for a career in {current_field}.
        
        User information:
        - Education Level: {education_level}
        - Prior Exposure to {current_field}: {prior_exposure}
        
        Instructions:
        1. Ask one question at a time about their skills, interests, and experience relevant to {current_field}.
        2. Make your questions conversational, engaging, and specific to different aspects of {current_field}.
        3. Customize your questions based on their education level ({education_level}) and prior exposure ({prior_exposure}).
        4. You are free to ask as many questions as you need to make a thorough assessment. Use your judgment to determine when you have enough information.
        5. Ensure your questions cover various aspects like aptitude, interest, relevant experience, and personality fit.
        6. Each question should be concise and direct - aim for 1-3 sentences per question.
        7. Ask your first question directly without an introduction.
        8. When you feel you have enough information to make a proper assessment, end your response with the text "ASSESSMENT_READY".
        """
        
        # Send the initial prompt to set up the context
        chat_session.send_message(initial_prompt)
        
        # Get the first question
        first_q_prompt = "Please ask your first question to assess the user's suitability for this field, considering their education level and prior exposure."
        response = chat_session.send_message(first_q_prompt)
        question_count += 1
        
        return jsonify({
            "response": response.text, 
            "response_type": "question",
            "assessment_complete": False,
            "question_number": question_count,
        })
    except Exception as e:
        print(f"Assessment initialization error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Assessment initialization error: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global chat_session, current_field, question_count, assessment_in_progress
    
    try:
        if not gemini_available:
            return jsonify({"error": "Gemini API is not available. Check server logs."}), 500
            
        if not chat_session or not current_field or not assessment_in_progress:
            return jsonify({
                "response": "Please start the assessment first by selecting a field.", 
                "response_type": "error",
                "assessment_complete": False
            }), 400

        # Get data from request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
            
        # Get the user's message
        user_message = data['message']
        
        # Send user message and include instructions for next question in the same prompt
        # Removed references to min_questions and max_questions constraints
        combined_prompt = f"""
        User response: {user_message}

        Based on this response, please assess the information provided and then ask the next appropriate question 
        to continue evaluating the user's suitability for {current_field}.
        
        Remember:
        1. Make your question conversational and relevant to a different aspect of the field than previously covered.
        2. This is question #{question_count + 1}. You've asked {question_count} questions so far.
        3. You are free to continue asking questions until you feel you have enough information to make a thorough assessment.
        4. Use your judgment to decide when you have sufficient information about the user's skills, interests, and aptitude.
        5. When you feel you have enough information to make a proper assessment, end your response with the text "ASSESSMENT_READY".
        """
        
        # Safety check to prevent infinite questioning (server-side protection)
        if question_count >= absolute_max_questions:
            combined_prompt += "\n\nYou have asked many questions already. Please conclude your assessment after this question by adding ASSESSMENT_READY to your response."
        
        # Send message to LLM - only one API call
        response = chat_session.send_message(combined_prompt)
        question_count += 1
        
        # Check if the LLM indicated it's ready for final assessment
        # or if we've reached absolute max questions
        assessment_ready = "ASSESSMENT_READY" in response.text or question_count >= absolute_max_questions
        
        if assessment_ready:
            # Clean the response if needed
            cleaned_response = response.text.replace("ASSESSMENT_READY", "").strip()
            
            # Generate final assessment
            final_prompt = f"""
            Based on all the responses from the user, provide a comprehensive assessment of their suitability for a career in {current_field}.
            
            Your assessment must follow this exact structure with these exact headings and use JSON format:

            Return ONLY a valid JSON object with these exact keys and nothing else:
            {{
                "suitability_score": "A number from 0-100 representing percentage match",
                "strengths": ["strength1", "strength2", "strength3", ...],
                "areas_for_improvement": ["area1", "area2", "area3", ...],
                "recommendation": "YES or NO",
                "recommendation_reason": "Brief explanation of your recommendation",
                "alternative_fields": ["field1", "field2", "field3"],
                "next_steps": ["step1", "step2", "step3"]
            }}

            Make sure your response is valid JSON that can be parsed. No markdown, no explanations outside the JSON.
            """
            
            assessment_response = chat_session.send_message(final_prompt)
            assessment_in_progress = False
            
            # Extract JSON from the response
            try:
                import json
                import re
                
                # Find JSON pattern in the response
                json_match = re.search(r'({[\s\S]*})', assessment_response.text)
                if json_match:
                    assessment_json = json.loads(json_match.group(1))
                    
                    # Return structured assessment data
                    return jsonify({
                        "response_type": "final_assessment",
                        "assessment_complete": True,
                        "field": current_field,
                        "assessment_data": {
                            "suitability_score": assessment_json.get("suitability_score", "N/A"),
                            "strengths": assessment_json.get("strengths", []),
                            "areas_for_improvement": assessment_json.get("areas_for_improvement", []),
                            "recommendation": assessment_json.get("recommendation", "N/A"),
                            "recommendation_reason": assessment_json.get("recommendation_reason", "N/A"),
                            "alternative_fields": assessment_json.get("alternative_fields", []),
                            "next_steps": assessment_json.get("next_steps", [])
                        },
                        # Include full text as backup
                        "full_response": assessment_response.text
                    })
                else:
                    # Fallback if JSON extraction fails
                    return jsonify({
                        "response": assessment_response.text,
                        "response_type": "final_assessment",
                        "assessment_complete": True,
                        "field": current_field,
                        "parsing_error": "Could not parse structured data from response"
                    })
            except Exception as e:
                # Fallback if JSON processing fails
                return jsonify({
                    "response": assessment_response.text,
                    "response_type": "final_assessment",
                    "assessment_complete": True,
                    "field": current_field,
                    "parsing_error": str(e)
                })
        else:
            # Return the response directly without sending another prompt
            return jsonify({
                "response": response.text,
                "response_type": "question",
                "assessment_complete": False,
                "question_number": question_count
            })
    except Exception as e:
        print(f"Chat error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Chat error: {str(e)}"}), 500
    

# -------------------- File Upload Route --------------------#
# Load SpaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model isn't downloaded, provide instructions
    print("Please download the SpaCy model with: python -m spacy download en_core_web_sm")
    nlp = None

# Configure upload settings
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX files"""
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file_path):
    """Extract text based on file type"""
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    else:
        return None

def parse_resume(text):
    """Parse resume text to extract structured information"""
    if not text:
        return {"error": "Could not extract text from document"}
    
    # Basic structure to store extracted information
    resume_data = {
        "contact_info": {},
        "education": [],
        "work_experience": [],
        "skills": [],
        "raw_text": text[:1000] + "..." if len(text) > 1000 else text  # Truncated raw text
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        resume_data["contact_info"]["email"] = emails[0]
    
    # Extract phone number
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        resume_data["contact_info"]["phone"] = phones[0]
    
    # Use SpaCy for named entity recognition if available
    if nlp:
        doc = nlp(text[:500000])  # Limit text length for processing
        
        # Extract skills - look for technical terms and keywords
        skill_keywords = ["Python", "JavaScript", "Java", "C++", "React", "Node.js", 
                          "SQL", "Machine Learning", "Data Analysis", "Project Management",
                          "Microsoft Office", "Leadership", "Communication", "AWS", "Docker",
                          "Kubernetes", "Git", "REST API", "HTML", "CSS", "Excel", "Marketing",
                          "Sales", "Customer Service", "Research", "Analysis", "Design"]
        
        for keyword in skill_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                resume_data["skills"].append(keyword)
        
        # Extract education information - basic pattern matching
        education_patterns = [
            r'(?:bachelor|master|phd|doctor|associate|b\.s\.|m\.s\.|b\.a\.|m\.a\.|ph\.d\.)\s+(?:of|in|degree)?\s+([^,\n]+)',
            r'(?:university|college|institute|school) of ([^,\n]+)',
        ]
        
        for pattern in education_patterns:
            education_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in education_matches:
                if match and match.strip() and len(match) < 100:  # Sanity check on match length
                    resume_data["education"].append(match.strip())
        
        # Extract potential companies/organizations
        for ent in doc.ents:
            if ent.label_ == "ORG" and len(ent.text) > 2:
                # Check if it looks like a company
                if not any(term.lower() in ent.text.lower() for term in ["university", "college", "school"]):
                    resume_data["work_experience"].append(ent.text)
        
        # Remove duplicates
        resume_data["education"] = list(set(resume_data["education"]))[:5]  # Limit to top 5
        resume_data["work_experience"] = list(set(resume_data["work_experience"]))[:5]  # Limit to top 5
        resume_data["skills"] = list(set(resume_data["skills"]))[:15]  # Limit to top 15
    
    return resume_data

@app.route('/resume-assessment', methods=['POST'])
def resume_assessment():
    try:
        # Check if Gemini is available
        if not gemini_available:
            return jsonify({"error": "Gemini API is not available. Check server logs."}), 500
            
        # Check if a file was uploaded
        if 'resume' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        # Check if target field was provided
        if 'field' not in request.form:
            return jsonify({"error": "Target field is required"}), 400
            
        file = request.files['resume']
        field = request.form['field']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save and process the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract text from file
            text = extract_text_from_file(file_path)
            
            # Parse the resume
            resume_data = parse_resume(text)
            
            # Delete the temporary file
            os.remove(file_path)
            
            # Prepare resume summary for Gemini
            resume_summary = f"""
            Field of interest: {field}
            
            Resume Summary:
            - Contact: {resume_data.get('contact_info', {})}
            - Education: {', '.join(resume_data.get('education', [])[:3]) or 'Not specified'}
            - Experience: {', '.join(resume_data.get('work_experience', [])[:3]) or 'Not specified'}
            - Skills: {', '.join(resume_data.get('skills', [])[:10]) or 'Not specified'}
            
            Raw Text Extract:
            {resume_data.get('raw_text', 'No text available')[:2000]}
            """
            
            # Send to Gemini for direct assessment
            assessment_prompt = f"""
            You are a career assessment specialist. Based on the resume information below, evaluate this person's 
            suitability for a career in {field}.
            
            {resume_summary}
            
            Analyze this resume information and assess suitability for {field}. Return ONLY a valid JSON object with these exact keys:
            {{
                "suitability_score": "A number from 0-100 representing percentage match",
                "strengths": ["strength1", "strength2", "strength3", ...],
                "areas_for_improvement": ["area1", "area2", "area3", ...],
                "recommendation": "YES or NO",
                "recommendation_reason": "Brief explanation of your recommendation",
                "alternative_fields": ["field1", "field2", "field3"],
                "next_steps": ["step1", "step2", "step3"]
            }}
            
            Make sure your response is valid JSON that can be parsed. No markdown, no explanations outside the JSON.
            """
            
            # Create a one-time chat for this assessment
            assessment_chat = gemini_model.start_chat(history=[])
            assessment_response = assessment_chat.send_message(assessment_prompt)
            
            # Extract JSON from the response
            try:
                import json
                import re
                
                # Find JSON pattern in the response
                json_match = re.search(r'({[\s\S]*})', assessment_response.text)
                if json_match:
                    assessment_json = json.loads(json_match.group(1))
                    
                    # Return structured assessment data
                    return jsonify({
                        "response_type": "resume_assessment",
                        "field": field,
                        "resume_data": resume_data,
                        "assessment_data": {
                            "suitability_score": assessment_json.get("suitability_score", "N/A"),
                            "strengths": assessment_json.get("strengths", []),
                            "areas_for_improvement": assessment_json.get("areas_for_improvement", []),
                            "recommendation": assessment_json.get("recommendation", "N/A"),
                            "recommendation_reason": assessment_json.get("recommendation_reason", "N/A"),
                            "alternative_fields": assessment_json.get("alternative_fields", []),
                            "next_steps": assessment_json.get("next_steps", [])
                        }
                    })
                else:
                    # Fallback if JSON extraction fails
                    return jsonify({
                        "response": assessment_response.text,
                        "response_type": "resume_assessment",
                        "field": field,
                        "resume_data": resume_data,
                        "parsing_error": "Could not parse structured data from response"
                    })
            except Exception as e:
                # Fallback if JSON processing fails
                return jsonify({
                    "response": assessment_response.text,
                    "response_type": "resume_assessment",
                    "field": field,
                    "resume_data": resume_data,
                    "parsing_error": str(e)
                })
                
        except Exception as e:
            # Delete the temporary file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Resume assessment error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Resume assessment error: {str(e)}"}), 500
    
# -------------------- Run App --------------------
if __name__ == '__main__':
    app.config['PROPAGATE_EXCEPTIONS'] = True  # Propagate exceptions to see detailed errors
    app.run(debug=True)