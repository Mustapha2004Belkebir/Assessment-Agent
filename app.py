from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
# Updated to use an available model from your list
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Changed from 'gemini-pro'

# Load Random Forest model
with open('random_forest.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Global state for Gemini chat
chat_session = None
current_field = None
assessment_in_progress = False
question_count = 0
max_questions = 15

# -------------------- Home --------------------
@app.route('/')
def index():
    return "Flask App: Random Forest API & Gemini Career Assessment"

# -------------------- Random Forest Route --------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load input data from frontend
        input_data = request.get_json()
        answers = input_data['answers']  # Expecting a dict
        
        # Convert to DataFrame
        data = pd.DataFrame([answers])  # One-row DataFrame

        # --- Encoding Section ---
        ordinal_columns = [
            "Do you enjoy and feel comfortable with subjects like mathematics, physics, and biology?",
            "Are you excited by combining theoretical learning with hands-on practical work?",
            "How do you handle long study hours and challenging academic content?",
            "How do you feel about public speaking or presenting?"
        ]
        ordinal_order = [
            [
                "I am very enthusiastic about these subjects and consistently excel in them.",
                "I find them interesting, although I sometimes face challenges.",
                "I can manage these subjects, but they aren't my favorite.",
                "I struggle with these subjects and do not feel very comfortable with them."
            ],
            [
                "I love the mix of theory and practice; it makes learning dynamic.",
                "I enjoy both but lean slightly toward theoretical work.",
                "I prefer hands-on practical work over extensive theory.",
                "I'm not enthusiastic about practical work and would rather stick to theory."
            ],
            [
                "I thrive under academic pressure.",
                "I can manage but find it exhausting.",
                "I prefer short, focused sessions.",
                "I dislike intense studying."
            ],
            [
                "I enjoy it and feel confident in front of an audience.",
                "I'm comfortable with it but prefer smaller groups.",
                "I find it stressful but can manage if necessary.",
                "I avoid it whenever possible."
            ]
        ]

        # Apply Ordinal Encoding
        ordinal_encoder = OrdinalEncoder(categories=ordinal_order)
        data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])

        # One-Hot Encode remaining categorical features
        categorical_columns = [col for col in data.columns if col not in ordinal_columns]
        onehot_encoder = OneHotEncoder(sparse_output=False)
        onehot_encoded = onehot_encoder.fit_transform(data[categorical_columns])
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))

        # Final input for model
        final_df = pd.concat([data[ordinal_columns], onehot_df], axis=1)

        # --- Prediction ---
        features = final_df.to_numpy()
        prediction = rf_model.predict(features)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------- Gemini Assessment Routes --------------------
@app.route('/start_assessment', methods=['POST'])
def start_assessment():
    global chat_session, current_field, assessment_in_progress, question_count
    
    try:
        # Get data from request
        data = request.get_json()
        if not data or 'field' not in data:
            return jsonify({"error": "Field is required"}), 400
            
        # Reset conversation state
        question_count = 0
        assessment_in_progress = True
        
        # Get the field directly from the request
        current_field = data['field']
        
        # Initialize chat session with empty history
        chat_session = gemini_model.start_chat(history=[])
        
        # Send system prompt as the first message
        initial_prompt = f"""
        You are an assessment assistant for the field of {current_field}. Your task is to assess if the user is suitable for a career in {current_field}.
        
        Instructions:
        1. Ask one question at a time about their skills, interests, and experience relevant to {current_field}.
        2. After receiving their answer, analyze it briefly, then ask the next question.
        3. Ask a total of {max_questions} questions that cover different aspects of {current_field}.
        4. Make your questions conversational and engaging.
        5. Start directly with your first question without any introduction.
        """
        
        response = chat_session.send_message(initial_prompt)
        
        # Ask LLM to generate the first question
        response = chat_session.send_message("Please ask your first question about my suitability for this field.")
        question_count += 1
        
        return jsonify({
            "response": response.text, 
            "assessment_complete": False,
            "question_number": question_count,
            "total_questions": max_questions
        })
    except Exception as e:
        return jsonify({"error": f"Assessment initialization error: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global chat_session, current_field, question_count, assessment_in_progress
    
    try:
        if not chat_session or not current_field or not assessment_in_progress:
            return jsonify({
                "response": "Please start the assessment first by selecting a field.", 
                "assessment_complete": False
            }), 400

        # Get data from request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
            
        # Get the user's message
        user_message = data['message']
        
        # Send message to LLM
        response = chat_session.send_message(user_message)
        
        # Check if we've reached the max number of questions
        if question_count >= max_questions:
            # Generate final assessment
            final_prompt = f"""
            Based on all the responses from the user, provide a comprehensive assessment of their suitability for a career in {current_field}.
            
            Your assessment should include:
            1. Strengths identified from their responses
            2. Areas for improvement or skills they might need to develop
            3. A clear recommendation on whether they should pursue this career path with a percentage match (e.g., "80% match with {current_field}")
            4. Additional advice for success in this field
            
            Format your response with clear headings for each section and provide specific details based on their responses.
            """
            
            response = chat_session.send_message(final_prompt)
            assessment_in_progress = False
            
            return jsonify({
                "response": response.text,
                "assessment_complete": True,
                "field": current_field
            })
        else:
            # Ask the next question
            next_prompt = f"""
            Thank you for your response. Please ask the next question to continue assessing the user's suitability for {current_field}.
            Remember to make it conversational and relevant to a different aspect of the field than previously covered.
            """
            
            response = chat_session.send_message(next_prompt)
            question_count += 1
            
            return jsonify({
                "response": response.text,
                "assessment_complete": False,
                "question_number": question_count,
                "total_questions": max_questions
            })
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500
    
# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(debug=True)