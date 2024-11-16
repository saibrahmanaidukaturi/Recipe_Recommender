import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit.components.v1 as components
import re
import json
import firebase_admin
from firebase_admin import credentials, firestore


# Get CSS path
def get_css_path():
    return os.path.join(os.path.dirname(__file__), 'styles.css')

@st.cache_data
def get_csv_path():
    return os.path.join(os.path.dirname(__file__),"..","data",'food.csv')

# Load and preprocess data
@st.cache_data
def load_data():
    csv_path = get_csv_path()
    return pd.read_csv(csv_path)

@st.cache_data
def preprocess_combined(df):
    df['TotalTimeInMins'] = pd.to_numeric(df['TotalTimeInMins'], errors='coerce').fillna(0).astype(int)
    df['combined'] = df['RecipeName'].astype(str) + " " + \
                     df["Ingredients"].astype(str) + " " + \
                     df["TotalTimeInMins"].astype(str) + " " + \
                     df['Cuisine'].astype(str) + " " + \
                     df['Course'].astype(str) + " " + \
                     df["Diet"].astype(str)
    return df['combined'].fillna('').str.lower()


@st.cache_data
def get_recommendations(fav_dish, df, num_recommendations=50):
    if 'combined' not in df.columns:
        df['combined'] = preprocess_combined(df)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    # Transform the user's input into a TF-IDF vector
    fav_dish_tfidf = vectorizer.transform([fav_dish])
    
    # Calculate cosine similarity between user input and all recipes
    cosine_sim = cosine_similarity(fav_dish_tfidf, tfidf_matrix)
    
    # Get indices of recipes sorted by similarity
    similar_indices = cosine_sim.argsort()[0, -num_recommendations:][::-1]
    
    return df.iloc[similar_indices]
    
@st.cache_data
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

@st.cache_data  
def search_recipes(df, cuisine, course, diet, max_total_time):
    results = df[df['TotalTimeInMins'] <= max_total_time]
    if cuisine != "Any":
        results = results[results['Cuisine'] == cuisine]
    if course != "Any":
        results = results[results['Course'] == course]
    if diet != "Any":
        results = results[results['Diet'] == diet]
    return results


@st.cache_data
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

def scroll_to_recipe_details():
    # Inject JavaScript to scroll to the `recipe-details` section
    components.html(
        """
        <script>
            const element = document.getElementById("recipe-details");
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        </script>
        """,
        height=0,  # Set height to 0 as we only need to run the script
    )


def display_search_results(results):
    # Initialize session state variables if they don't exist
    if "show_recipe_details" not in st.session_state:
        st.session_state.show_recipe_details = None
    if "page" not in st.session_state:
        st.session_state.page = 1

    # Display full recipe details before all results
    if st.session_state.show_recipe_details is not None:
        # Get the recipe based on the index stored in session state
        idx = st.session_state.show_recipe_details
        recipe = results.iloc[idx]
        
        # Display recipe details in an expander
        scroll_to_recipe_details()
        st.markdown('<a id="recipe-details"></a>', unsafe_allow_html=True)
        with st.expander("Recipe Details", expanded=True):
            show_recipe_details(recipe)
            if st.button("Close Recipe"):
                st.session_state.show_recipe_details = None
                st.rerun()

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
                # When the user clicks on a recipe, store the index in the session state
                if st.button("Show recipe", key=row['RecipeName'], use_container_width=True):
                    st.session_state.show_recipe_details = start_idx + idx
                    st.rerun()

        st.markdown("---")
        
        # Calculate total pages
        total_pages = (len(results) - 1) // 6 + 1

        # Display page navigation buttons in the same line
        col1, col2, col3 = st.columns([2, 8, 1])

        with col1:
            if st.session_state.page > 1:
                if st.button("< Previous"):
                    st.session_state.page -= 1
                    st.session_state.show_recipe_details = None  # Reset recipe details when page changes
                    st.rerun() 

        with col2:
            st.markdown(f'<div style="text-align: center;">Page {st.session_state.page} of {total_pages}</div>', unsafe_allow_html=True)

        with col3:
            if st.session_state.page < total_pages:
                if st.button("Next >"):
                    st.session_state.page += 1
                    st.session_state.show_recipe_details = None  # Reset recipe details when page changes


@st.cache_data
def display_recommendations(query):
    df = fetch_data("food")
    #df = load_data()
    recommendations = get_recommendations(query, df)
    return recommendations


@st.cache_data
def get_applied_filters(cuisine, course, diet, max_total_time):
    filters = []
    if cuisine != "Any":
        filters.append(f"Cuisine: {cuisine}")
    if course != "Any":
        filters.append(f"Course: {course}")
    if diet != "Any":
        filters.append(f"Diet: {diet}")
    if max_total_time != 360:
        filters.append(f"Max Total Time: {max_total_time} mins")
    return ", ".join(filters) if filters else "No filters applied."

@st.cache_data
# Fetch data from Firestore and convert to DataFrame
def fetch_data(collection_name):
    if not firebase_admin._apps:
        # Access the service account key from Streamlit secrets and load it as a dictionary
        service_account_key = st.secrets["firebase"]["service_account_key"]
        service_account_dict = json.loads(service_account_key)

        # Use the credentials dictionary to initialize Firebase
        cred = credentials.Certificate(service_account_dict)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    docs = db.collection(collection_name).stream()
    data = [doc.to_dict() for doc in docs]
    return pd.DataFrame(data)

# Function to tokenize user input
def tokenize_input(user_message):
    url = "https://api.arliai.com/v1/tokenize"
    ARLIAI_API_KEY = os.getenv('ARLIAI_API_KEY')
    
    payload = json.dumps({
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    })
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {ARLIAI_API_KEY}"
    }

    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        return response.json()  # Return the tokenized output
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None