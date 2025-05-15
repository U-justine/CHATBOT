from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load and prepare the data
def load_qa_data():
    # Replace 'qa_data.xlsx' with your Excel file name
    df = pd.read_excel(r'C:\Users\HP\PycharmProjects\PythonProject\ICT_QA.xlsx')
    return df

# Initialize the model
def initialize_model(df):
    vectorizer = TfidfVectorizer()
    questions_vectors = vectorizer.fit_transform(df['Question'])
    return vectorizer, questions_vectors

# Load data and initialize model
qa_df = load_qa_data()
vectorizer, questions_vectors = initialize_model(qa_df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    
    # Transform user message
    user_vector = vectorizer.transform([user_message])
    
    # Calculate similarity
    similarities = cosine_similarity(user_vector, questions_vectors)
    most_similar_idx = np.argmax(similarities[0])
    
    # Get the answer
    if similarities[0][most_similar_idx] > 0.3:  # Similarity threshold
        answer = qa_df.iloc[most_similar_idx]['Answer']
    else:
        answer = "I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing or contact the ICT department directly."
    
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)