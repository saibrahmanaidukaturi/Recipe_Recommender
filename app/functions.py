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
def get_recommendations(fav_dish, df, num_recommendations=50):
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

def search_recipes(df, cuisine, course, diet, max_total_time):
    results = df[df['TotalTimeInMins'] <= max_total_time]
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
    # Initialize session state variables if they don't exist
    if "show_recipe_details" not in st.session_state:
        st.session_state.show_recipe_details = None
    if "page" not in st.session_state:
        st.session_state.page = 1

    # Handle if there are no results
    if len(results) == 0:
        st.write("No recipes found. Try adjusting your filters.")
    else:
        # Calculate start and end index for the results on the current page
        start_idx = (st.session_state.page - 1) * 6
        end_idx = start_idx + 6
        page_results = results.iloc[start_idx:end_idx]

        # Create columns for displaying the results
        cols = st.columns(3)
        
        # Iterate over the page results (not the whole results)
        for idx, (_, row) in enumerate(page_results.iterrows()):
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
                # Allow expanding for recipe details
                with st.expander("See Recipe Details"):
                    # When the user clicks on a recipe, store the index in the session state
                    if st.button(f"Show Details for {row['RecipeName']}", key=row['RecipeName']):
                        st.session_state.show_recipe_details = idx

        if st.session_state.show_recipe_details is not None:
            idx = st.session_state.show_recipe_details
            recipe = page_results.iloc[idx]
            
            with st.expander("Recipe Details", expanded=True):
                show_recipe_details(recipe)
                if st.button("Close"):
                    st.session_state.show_recipe_details = None

        st.markdown("---")
        
        # Calculate total pages
        total_pages = (len(results) - 1) // 6 + 1

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


def display_recommendations(df, fav_dish):
    recommended_recipes = get_recommendations(fav_dish, df)
    return recommended_recipes