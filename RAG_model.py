import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai  # Ensure this is imported

# Load data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)  # Use the provided file path
    return df

# Create embeddings for the dataset using Universal Sentence Encoder
def create_embeddings(df, model):
    embeddings = model(df['text'].tolist())
    return embeddings

# Retrieve relevant information based on user query
def retrieve_info(query, df, embeddings, model, top_k=3):
    query_embedding = model([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]['text'].tolist()

# Generate response using Gemini
def generate_response(probability, query, relevant_info):
    genai.configure(api_key='#############################')  # Replace with your API key
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Given the following information:
    - Probability of Autism: {probability}%
    - User Query: {query}
    - Relevant Information:
    {' '.join(relevant_info)}

    Provide a comprehensive explanation of the person's current state regarding autism 
    and what they should do next. Be empathetic and informative in your response. keep your response short.
    """

    response = model.generate_content(prompt)
    return response.text

# Main function to run the RAG model
def run_rag_model(keras_probability, user_query, data_path='./autism-info-csv.csv'):
    # Convert Keras output (float) to percentage
    probability = keras_probability * 100  # Assuming keras_probability is between 0 and 1

    # Load data and create embeddings
    df = load_data(data_path)
    embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = create_embeddings(df, embedding_model)

    # Retrieve relevant information
    relevant_info = retrieve_info(user_query, df, embeddings, embedding_model)

    # Generate response
    response = generate_response(probability, user_query, relevant_info)

    return response

# Example usage
if __name__ == "__main__":
    keras_probability = 0.67  # Example output from the Keras model (0 to 1)
    user_query = "What should I do if I think I might have autism?"
    data_path = "./autism-info-csv.csv"  # Path to your CSV file

    response = run_rag_model(keras_probability, user_query, data_path="./autism-info-csv.csv")
    print(response)
