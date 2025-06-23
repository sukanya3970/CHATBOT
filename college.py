import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set Streamlit page config
st.set_page_config(page_title="SVECW College Chatbot 🎓", page_icon="🎓", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Choose CSV file ---
st.sidebar.title("🔍 Select FAQ Dataset")
csv_choice = st.sidebar.selectbox("Choose dataset:", ["college_faq.csv", "svcew_details.csv"])

# Map filenames to raw GitHub URLs
csv_urls = {
    "college_faq.csv": "https://raw.githubusercontent.com/sukanya3970/CHATBOT/main/college_faq.csv",
    "svcew_details.csv": "https://raw.githubusercontent.com/sukanya3970/CHATBOT/main/svcew_details.csv"
}

csv_url = csv_urls[csv_choice]

# --- Load CSV ---
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

# --- Preprocess ---
df = df.fillna("")
if 'Question' not in df.columns or 'Answer' not in df.columns:
    st.error("CSV must contain 'Question' and 'Answer' columns.")
    st.stop()

df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# --- Configure Gemini API ---
API_KEY = st.secrets["api_keys"]["gemini"]  # Add this to .streamlit/secrets.toml
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Find Closest Question ---
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:
        return df.iloc[best_match_index]['Answer']
    else:
        return None

# --- Streamlit Chat UI ---
st.title("SVECW College Chatbot 🎓")
st.write("Welcome! Ask me anything about SVECW college, academics, departments, or rules.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Check CSV for answer
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)

    if closest_answer:
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
        with st.chat_message("assistant"):
            st.markdown(closest_answer)
    else:
        # Step 2: Use Gemini if no good match
        try:
            response = model.generate_content(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Gemini failed to respond. Error: {e}")
