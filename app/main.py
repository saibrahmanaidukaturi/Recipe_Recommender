import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(page_title="Recipe Recommender", layout="wide")

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
cuisine = st.sidebar.selectbox("Select cuisine:", ["Any", "Italian", "Mexican", "Chinese", "Indian", "American"])
diet = st.sidebar.multiselect("Dietary restrictions:", ["Vegetarian", "Vegan", "Gluten-free", "Dairy-free"])

# Search button
if st.sidebar.button("Search Recipes"):
    st.write("Searching for recipes...")  # Placeholder for search functionality

# Main content
st.header("Featured Recipes")

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://via.placeholder.com/300x200.png?text=Recipe+1", use_column_width=True, caption="Recipe 1")
    st.write("Recipe 1 description")

with col2:
    st.image("https://via.placeholder.com/300x200.png?text=Recipe+2", use_column_width=True, caption="Recipe 2")
    st.write("Recipe 2 description")

with col3:
    st.image("https://via.placeholder.com/300x200.png?text=Recipe+3", use_column_width=True, caption="Recipe 3")
    st.write("Recipe 3 description")

# Footer
st.markdown("---")
st.write("© 2024 Recipe Recommender App")