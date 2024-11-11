import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Get CSS path
def get_css_path():
    return os.path.join(os.path.dirname(__file__), 'styles.css')

# Load and preprocess data
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

@st.cache_data
def preprocess_combined(df):
    df['combined'] = df['RecipeName'].astype(str) + " " + \
                     df["Ingredients"].astype(str) + " " + \
                     df["TotalTimeInMins"].astype(str) + " " + \
                     df['Cuisine'].astype(str) + " " + \
                     df['Course'].astype(str) + " " + \
                     df["Diet"].astype(str)
    return df['combined'].fillna('').str.lower()

@st.cache_resource
def create_tfidf_vectorizer(ingredients):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer, vectorizer.fit_transform(ingredients)

# Recommendation system
@st.cache_data
def get_recommendations(fav_dish, df, num_recommendations=5):
    if 'combined' not in df.columns:
        preprocess_combined(df)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    fav_dish_tfidf = vectorizer.transform([fav_dish])
    cosine_sim = cosine_similarity(fav_dish_tfidf, tfidf_matrix)
    similar_indices = cosine_sim.argsort()[0, -num_recommendations:][::-1]
    return df.iloc[similar_indices]

def scrape_recipe_image(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('div', class_='recipe-image').find('img')
        if img_tag and 'src' in img_tag.attrs:
            return urljoin(url, img_tag['src'])
    except Exception as e:
        print(f"Error scraping image from {url}: {e}")
    return None

def search_recipes(df, recipe_name, ingredient, cuisine, course, diet, max_total_time):
    results = df[df['TotalTimeInMins'] <= max_total_time]
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

# Display functions
def show_recipe_details(recipe):
    st.subheader(recipe['RecipeName'])
    st.image(scrape_recipe_image(recipe['URL']), width=300)
    st.write(f"**Ingredients:** {recipe['Ingredients']}")
    st.write(f"**Total Time:** {recipe['TotalTimeInMins']} minutes")
    st.write(f"**Cuisine:** {recipe['Cuisine']}, **Course:** {recipe['Course']}, **Diet:** {recipe['Diet']}")
    st.subheader("Instructions:")
    st.write(recipe['Instructions'])
    st.markdown(f"[View full recipe]({recipe['URL']})")

def display_search_results(results):
    if len(results) == 0:
        st.write("No recipes found. Try adjusting your search criteria.")
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                st.markdown(f'''
                    <div class="recipe-card">
                    <img src="{scrape_recipe_image(row['URL'])}" class="recipe-image">
                    <div class="recipe-title">{row['RecipeName']}</div>
                    <div class="recipe-info">
                        Cuisine: {row['Cuisine']}<br>
                        Course: {row['Course']}<br>
                        Total Time: {row['TotalTimeInMins']} minutes
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                with st.expander("See Recipe Details"):
                    show_recipe_details(row)

def display_recommendations(df, fav_dish):
    recommended_recipes = get_recommendations(fav_dish, df)
    return recommended_recipes