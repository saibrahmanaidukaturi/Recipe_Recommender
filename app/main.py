import streamlit as st
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="AI Recipe Recommender", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'food.csv')

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

@st.cache_data
def preprocess_ingredients(df):
    return df['Ingredients'].fillna('').str.lower()

@st.cache_resource
def create_tfidf_vectorizer(ingredients):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(ingredients)
    return vectorizer, tfidf_matrix

# CSS for consistent image sizing and card layout
st.markdown("""
    <style>
        .recipe-card {
            width: 300px; 
            height: 400px; 
            background-color: #f9f9f9; 
            border-radius: 10px; 
            padding: 10px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            justify-content: space-between; 
            overflow: hidden;  
        }

        .recipe-image {
            width: 100%;  
            height: 200px; 
            object-fit: cover; 
            margin-bottom: 10px;  
            border-radius: 8px;  
        }

        .recipe-title {
            font-family: Arial, sans-serif;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
            text-align: center;  
            white-space: nowrap;  
            overflow: hidden;     
            text-overflow: ellipsis;  
        }

        .recipe-info {
            font-family: Arial, sans-serif;
            font-size: 14px;
            text-align: center;
            margin-bottom: 10px;
        }

        .recipe-card a {
            text-align: center;
            font-size: 14px;
            color: #0066cc;
            text-decoration: none;
        }

        .recipe-card a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

def get_recommendations(recipe_index, cosine_sim, df, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    recipe_indices = [i[0] for i in sim_scores]
    return df.iloc[recipe_indices]

def scrape_recipe_image(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        recipe_image_div = soup.find('div', class_='recipe-image')
        if recipe_image_div:
            img_tag = recipe_image_div.find('img')
            if img_tag and 'src' in img_tag.attrs:
                img_url = img_tag['src']
                img_url = urljoin(url, img_url)
                return img_url
        return None
    except Exception as e:
        print(f"Error scraping image from {url}: {e}")
        return None

# Title
st.markdown(f'<div style="text-align: center;"><h1>🍳 AI Recipe Recommender</h1></div>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state if not already initialized
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()

if 'page' not in st.session_state:
    st.session_state.page = 1

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df = load_data()

# Preprocess ingredients and create the TF-IDF vectorizer
ingredients = preprocess_ingredients(df)
vectorizer, tfidf_matrix = create_tfidf_vectorizer(ingredients)

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Set default values
default_recipe_name = ""  # Default is empty
default_ingredient = ""  # Default is empty
default_cuisine = "Any"  # Default is 'Any'
default_course = "Any"  # Default is 'Any'
default_diet = "Any"  # Default is 'Any'
default_max_time = 60  # Default is 60 minutes

# Sidebar for search options
with st.sidebar:
    st.header("Search Options")
    recipe_name = st.text_input("Search by recipe name:")
    ingredient = st.text_input("Enter main ingredient:")
    cuisine = st.selectbox("Select cuisine:", ["Any"] + sorted(df['Cuisine'].unique().tolist()))
    course = st.selectbox("Select course:", ["Any"] + sorted(df['Course'].unique().tolist()))
    diet = st.selectbox("Select diet:", ["Any"] + sorted(df['Diet'].unique().tolist()))
    max_total_time = st.slider("Maximum time (minutes):", 0, 120, 60)
    search_button = st.button("Search Recipes", key="search")

def search_recipes(recipe_name, ingredient, cuisine, course, diet, max_total_time):
    results = df[(df['TotalTimeInMins'] <= max_total_time)]
    if recipe_name:
        results = results[results['RecipeName'].str.contains(recipe_name, case=False, na=False)]
    if ingredient:
        results = results[results['Ingredients'].str.contains(ingredient, case=False, na=False)]
    if cuisine != "Any":
        results = results[results['Cuisine'] == cuisine]
    if course != "Any":
        results = results[results['Course'] == course]
    if diet != "Any":
        results = results[results['Diet'] == diet]
    return results

# Search and display results
if search_button:
    st.session_state.results = search_recipes(recipe_name, ingredient, cuisine, course, diet, max_total_time)
    st.session_state.page = 1  # Reset to first page after a new search

results = st.session_state.results  # Get the search results from session state

if len(results) == 0:
    st.write("No recipes found. Try adjusting your search criteria.")
else:
    # Calculate start and end index for the results on the current page
    start_idx = (st.session_state.page - 1) * 6
    end_idx = start_idx + 6
    page_results = results.iloc[start_idx:end_idx]

    # Show the results for the current page
    cols = st.columns(3)
    for idx, recipe in enumerate(page_results.iterrows()):
        _, row = recipe
        with cols[idx % 3]:  # Cycle through the columns for each recipe
            st.markdown(f'''
                <div class="recipe-card">
                    <img src="{scrape_recipe_image(row['URL'])}" class="recipe-image">
                    <div class="recipe-title">{row['RecipeName']}</div>
                    <div class="recipe-info">
                        Cuisine: {row['Cuisine']}<br>
                        Course: {row['Course']}<br>
                        Total Time: {row['TotalTimeInMins']} minutes
                    </div>
                    <a href="{row['URL']}" target="_blank">View full recipe</a>
                </div>
            ''', unsafe_allow_html=True)
    st.markdown("---")
        # Calculate total pages
    total_pages = (len(results) // 6) + (1 if len(results) % 6 != 0 else 0)

    # Display page navigation buttons in the same line
    col1, col2, col3 = st.columns([2, 8, 1])

    with col1:
        if st.session_state.page > 1:
            if st.button("< Previous"):
                st.session_state.page -= 1

    with col2:
        st.markdown(f'<div style="text-align: center;">Page {st.session_state.page} of {total_pages}</div>', unsafe_allow_html=True)

    with col3:
        if st.session_state.page < total_pages:
            if st.button("Next >"):
                st.session_state.page += 1


# AI-based Recommendations
st.markdown("---")
st.markdown("## Recommended Recipes")
random_recipe_index = df.sample(1).index[0]
recommended_recipes = get_recommendations(random_recipe_index, cosine_sim, df, num_recommendations=3)
cols = st.columns(3)
for idx, recipe in enumerate(recommended_recipes.iterrows()):
    _, row = recipe
    with cols[idx]:
        st.markdown(f'''
            <div class="recipe-card">
                <img src="{scrape_recipe_image(row['URL'])}" class="recipe-image">
                <div class="recipe-title">{row['RecipeName']}</div>
                <div class="recipe-info">
                    Cuisine: {row['Cuisine']}<br>
                    Course: {row['Course']}<br>
                    Total Time: {row['TotalTimeInMins']} minutes
                </div>
                <a href="{row['URL']}" target="_blank">View full recipe</a>
            </div>
        ''', unsafe_allow_html=True)
