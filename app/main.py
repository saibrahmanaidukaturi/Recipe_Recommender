import streamlit as st
import pandas as pd
from PIL import Image
import os


# Set page config
st.set_page_config(page_title="Recipe Recommender", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'food.csv')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)
df = load_data()

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
st.title("🍳 Recipe Recommender")

# Sidebar
st.sidebar.header("Search Options")
ingredient = st.sidebar.text_input("Enter main ingredient:")
cuisine = st.sidebar.selectbox("Select cuisine:", ["Any"] + sorted(df['Cuisine'].unique().tolist()))
course = st.sidebar.selectbox("Select course:", ["Any"] + sorted(df['Course'].unique().tolist()))
diet = st.sidebar.selectbox("Select diet:", ["Any"] + sorted(df['Diet'].unique().tolist()))
max_prep_time = st.sidebar.slider("Maximum preparation time (minutes):", 0, 120, 60)

# Search function
def search_recipes(ingredient, cuisine, course, diet, max_prep_time):
    results = df[
        (df['Ingredients'].str.contains(ingredient, case=False, na=False)) &
        (df['PrepTimeInMins'] <= max_prep_time)
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
    results = search_recipes(ingredient, cuisine, course, diet, max_prep_time)
    
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

# Footer
st.markdown("---")
st.write("© 2024 Recipe Recommender App")