import streamlit as st
import pandas as pd
from PIL import Image
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="AI Recipe Recommender", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'food.csv')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df = load_data()

# Preprocess the ingredients
@st.cache_data
def preprocess_ingredients(df):
    return df['Ingredients'].fillna('').str.lower()

# Create TF-IDF vectorizer
@st.cache_resource
def create_tfidf_vectorizer(ingredients):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(ingredients)
    return vectorizer, tfidf_matrix

# Get recipe recommendations
def get_recommendations(recipe_index, cosine_sim, df, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    recipe_indices = [i[0] for i in sim_scores]
    return df.iloc[recipe_indices]

# Prepare data for AI recommendations
ingredients = preprocess_ingredients(df)
vectorizer, tfidf_matrix = create_tfidf_vectorizer(ingredients)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .recipe-image {
        width: 100%;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🍳 AI Recipe Recommender")

# Sidebar
st.sidebar.header("Search Options")
ingredient = st.sidebar.text_input("Enter main ingredient:")
cuisine = st.sidebar.selectbox("Select cuisine:", ["Any"] + sorted(df['Cuisine'].unique().tolist()))
course = st.sidebar.selectbox("Select course:", ["Any"] + sorted(df['Course'].unique().tolist()))
diet = st.sidebar.selectbox("Select diet:", ["Any"] + sorted(df['Diet'].unique().tolist()))
max_total_time = st.sidebar.slider("Maximum time (minutes):", 0, 120, 60)

# Search function
def search_recipes(ingredient, cuisine, course, diet, max_total_time):
    results = df[
        (df['Ingredients'].str.contains(ingredient, case=False, na=False)) &
        (df['TotalTimeInMins'] <= max_total_time)
    ]
    
    if cuisine != "Any":
        results = results[results['Cuisine'] == cuisine]
    if course != "Any":
        results = results[results['Course'] == course]
    if diet != "Any":
        results = results[results['Diet'] == diet]
    
    return results

# Search button
if st.sidebar.button("Search Recipes"):
    results = search_recipes(ingredient, cuisine, course, diet, max_total_time)
    
    if len(results) == 0:
        st.write("No recipes found. Try adjusting your search criteria.")
    else:
        st.write(f"Found {len(results)} recipes:")
        for i, row in results.iterrows():
            with st.expander(f"{row['RecipeName']} ({row['Cuisine']} {row['Course']})"):
                st.write(f"Preparation Time: {row['PrepTimeInMins']} minutes")
                st.write(f"Cooking Time: {row['CookTimeInMins']} minutes")
                st.write(f"Total Time: {row['TotalTimeInMins']} minutes")
                st.write(f"Servings: {row['Servings']}")
                st.write(f"Diet: {row['Diet']}")
                st.write("Ingredients:")
                st.write(row['Ingredients'])
                st.write("Instructions:")
                st.write(row['Instructions'])
                st.write(f"[View full recipe]({row['URL']})")
                
                # AI Recommendations
                st.write("You might also like:")
                recommendations = get_recommendations(i, cosine_sim, df)
                for _, rec in recommendations.iterrows():
                    st.write(f"- {rec['RecipeName']} ({rec['Cuisine']} {rec['Course']})")

# Main content
st.header("Featured Recipes")

# Display 3 random recipes
featured_recipes = df.sample(3)
col1, col2, col3 = st.columns(3)

for i, (_, recipe) in enumerate(featured_recipes.iterrows()):
    with [col1, col2, col3][i]:
        st.subheader(recipe['RecipeName'])
        st.image("https://via.placeholder.com/300x200.png?text=Recipe+Image", use_column_width=True, caption=recipe['RecipeName'])
        st.write(f"Cuisine: {recipe['Cuisine']}")
        st.write(f"Course: {recipe['Course']}")
        st.write(f"Total Time: {recipe['TotalTimeInMins']} minutes")
        if st.button(f"View Recipe {i+1}"):
            st.write(f"Ingredients: {recipe['Ingredients']}")
            st.write(f"Instructions: {recipe['Instructions']}")
            st.write(f"[View full recipe]({recipe['URL']})")
            
            # AI Recommendations for featured recipes
            st.write("You might also like:")
            recommendations = get_recommendations(recipe.name, cosine_sim, df)
            for _, rec in recommendations.iterrows():
                st.write(f"- {rec['RecipeName']} ({rec['Cuisine']} {rec['Course']})")

# Footer
st.markdown("---")
st.write("© 2024 AI Recipe Recommender App")