import random
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
import logging

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(
    page_title="AI Health Assistant", 
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


import google.generativeai as genai

# Configure Gemini API
def configure_gemini(api_key):
    genai.configure(api_key=api_key)

# Initialize Gemini model
def initialize_gemini_model():
    return genai.GenerativeModel('gemini-pro')

# Function to get Gemini's response
def get_gemini_response(model, prompt):
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("AI-Powered Medical Chatbot 🤖")

# Input for Gemini API key
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
if not api_key:
    st.warning("Please enter your Gemini API key to proceed.")
    st.stop()

# Configure Gemini
configure_gemini(api_key)

# Initialize Gemini model
Gemini_model = initialize_gemini_model()

# Define refinement prompts
medical_advice_prompt = """
You are a knowledgeable and empathetic medical assistant. Your goal is to refine and improve the following medical advice response. Ensure the response is:
- Clear and easy to understand.
- Empathetic and supportive.
- Accurate and actionable.
- Includes a recommendation to consult a healthcare professional if the symptoms are serious or persistent.

Here is the original response:
"{original_response}"

Please refine it:
"""

health_tips_prompt = """
You are a friendly and motivational health coach. Your goal is to refine and improve the following health tip. Ensure the response is:
- Positive and encouraging.
- Actionable and practical.
- Tailored to the user's input (if provided).
- Includes a general health reminder if no specific input is given.

Here is the original response:
"{original_response}"

Please refine it:
"""

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
    }
    .css-1v0mbdj.e115fcil1 {
        max-width: 1200px;
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png")
    st.header("Settings")
    language_choice = st.selectbox("Select Language", [
        "English", "Hindi", "Gujarati", "Korean", "Turkish",
        "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese"
    ])
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This AI Health Assistant helps you with:
        - Medical advice
        - Health tips
        - Symptom analysis
    """)

# Main content
st.title("🏥 AI Health Assistant")
st.markdown("*Your personal health companion powered by AI*")

# Initialize components
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    try:
        return pd.read_csv(r'C:\Users\polar\Desktop\AI-Health-Assistant-main\dataset.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

model = load_model()
df = load_data()
translator = Translator()

# Medical keywords for fallback responses
medical_keywords = {
    "fever": "You might have a fever. Stay hydrated, rest, and consult a doctor if it exceeds 100.4°F (38°C) for over 24 hours.",
    "cough": "A cough could indicate an infection or allergy. Try warm tea with honey and rest; see a doctor if it persists.",
    "headache": "Headaches can stem from dehydration, stress, or tension. Rest, hydrate, and consider pain relief if needed.",
    "cold": "A cold typically resolves in 7-10 days. Rest, hydrate, and use saline sprays for relief.",
}

# Health tips dictionary
health_tips = {
    "sleep": [
        "Aim for 7-9 hours of sleep nightly for optimal health.",
        "Maintain a consistent sleep schedule to regulate your body clock.",
        "Limit screen time 1 hour before bed to improve sleep quality.",
    ],
    "energy": [
        "Eat balanced meals with protein, carbs, and healthy fats for sustained energy.",
        "Engage in 20-30 minutes of moderate exercise daily to boost energy.",
        "Drink 8-10 glasses of water daily to combat fatigue.",
    ],
    "stress": [
        "Practice deep breathing or meditation for 5-10 minutes to reduce stress.",
        "Take short breaks every hour to recharge mentally.",
        "Try yoga or stretching to relieve physical and mental tension.",
    ],
    "general": [
        "Incorporate colorful fruits and vegetables into your diet for essential nutrients.",
        "Stay active with at least 150 minutes of moderate exercise weekly.",
        "Schedule regular health checkups to monitor your well-being.",
    ],
}

# Function to analyze symptoms and suggest cures
def find_best_cure(user_input):
    if not model or df.empty:
        return "Error: Unable to process due to missing model or dataset."
    
    try:
        user_input_embedding = model.encode(user_input, convert_to_tensor=True)
        disease_embeddings = model.encode(df['disease'].tolist(), convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_input_embedding, disease_embeddings)[0]
        best_match_idx = similarities.argmax().item()
        best_match_score = similarities[best_match_idx].item()

        SIMILARITY_THRESHOLD = 0.6  # Increased for better precision
        if best_match_score >= SIMILARITY_THRESHOLD:
            return df.iloc[best_match_idx]['cure']
        
        # Fallback to keyword matching
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response
        
        return "I couldn’t identify your condition. Please provide more details or consult a healthcare professional."
    except Exception as e:
        logging.error(f"Error in find_best_cure: {e}")
        return "An error occurred while processing your request."

# Function to translate text with error handling
def translate_text(text, dest_language='en'):
    try:
        return translator.translate(text, dest=dest_language).text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Function to get personalized health tip
def get_personalized_health_tip(user_input):
    user_input_lower = user_input.lower()
    if any(word in user_input_lower for word in ["tired", "fatigue", "low energy"]):
        return random.choice(health_tips["energy"])
    elif any(word in user_input_lower for word in ["sleep", "rest", "insomnia"]):
        return random.choice(health_tips["sleep"])
    elif any(word in user_input_lower for word in ["stress", "anxiety", "tense"]):
        return random.choice(health_tips["stress"])
    return random.choice(health_tips["general"])  # Default general tip


# Language codes
language_codes = {
    "English": "en", "Hindi": "hi", "Gujarati": "gu", "Korean": "ko", "Turkish": "tr",
    "German": "de", "French": "fr", "Arabic": "ar", "Urdu": "ur", "Tamil": "ta",
    "Telugu": "te", "Chinese": "zh-CN", "Japanese": "ja"
}


# Function to refine response using Gemini
def refine_response(model, prompt_template, original_response):
    # Format the prompt with the original response
    prompt = prompt_template.format(original_response=original_response)
    # Get Gemini's refined response
    refined_response = model.generate_content(prompt)
    return refined_response.text

# Main chat interface
with st.container():
    st.markdown("### How can I help you today?")
    user_input = st.text_area(
        "Describe your symptoms or ask a health-related question:",
        height=100,
        placeholder="Example: I have a headache and fever..."
    )

    # col1, col2 = st.columns(2)
    
    # with col1:
    if st.button("🔍 Get Medical Advice", use_container_width=True):
        if user_input:
            with st.spinner('Analyzing your symptoms...'):
                try:
                    nlp_response = find_best_cure(user_input)
                    # Combine system prompt and user input for medical advice
                    response = refine_response(Gemini_model, medical_advice_prompt, nlp_response)
                    translated_response = translate_text(
                        response, 
                        dest_language=language_codes[language_choice]
                    )
                    st.success(f"{translated_response}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please describe your symptoms first.")

# with col2:
    if st.button("💡 Get Health Tips", use_container_width=True):
        if user_input:
            with st.spinner('Generating personalized tips...'):
                try:
                    nlp_response = get_personalized_health_tip(user_input)
                    # Combine system prompt and user input for health tips
                    response = refine_response(Gemini_model, health_tips_prompt, nlp_response)
                    translated_tip = translate_text(
                        response, 
                        dest_language=language_codes[language_choice]
                    )
                    st.info(f"{translated_tip}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter your query first.")

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if len(st.session_state.chat_history) > 0:
    st.markdown("### Previous Conversations")
    for i, (q, a) in enumerate(st.session_state.chat_history[-5:]):
        st.text_area(f"Question {i+1}", q, height=50, disabled=True)
        st.text_area(f"Answer {i+1}", a, height=100, disabled=True)
        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>⚠️ This is an AI assistant and should not replace professional medical advice. 
        If you have serious health concerns, please consult a healthcare professional.</small>
        <br>
        <small>Note: Translations are AI-powered and may not be perfect.</small>
    </div>
""", unsafe_allow_html=True)