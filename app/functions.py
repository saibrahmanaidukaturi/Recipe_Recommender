# functions.py
import os
import re
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

import streamlit as st
import streamlit.components.v1 as components

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# Paths / data
# ---------------------------
def get_css_path():
    return os.path.join(os.path.dirname(__file__), "styles.css")


@st.cache_data(show_spinner=False)
def get_csv_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "food.csv")


@st.cache_data(show_spinner=False)
def load_data():
    csv_path = get_csv_path()
    df = pd.read_csv(csv_path)

    # Normalize columns defensively
    for col in ["RecipeName", "Ingredients", "TotalTimeInMins", "Cuisine", "Course", "Diet", "Instructions", "URL"]:
        if col not in df.columns:
            df[col] = ""

    df["TotalTimeInMins"] = pd.to_numeric(df["TotalTimeInMins"], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def preprocess_combined(df: pd.DataFrame) -> pd.Series:
    combined = (
        df["RecipeName"].astype(str)
        + " "
        + df["Ingredients"].astype(str)
        + " "
        + df["TotalTimeInMins"].astype(str)
        + " "
        + df["Cuisine"].astype(str)
        + " "
        + df["Course"].astype(str)
        + " "
        + df["Diet"].astype(str)
    )
    return combined.fillna("").str.lower()


# ---------------------------
# BERT model / embeddings
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    # Optional device move (CPU by default; if you want GPU, uncomment next 2 lines)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    return tokenizer, model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Mask-aware mean pooling
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_bert_embeddings(texts, tokenizer, model, batch_size: int = 32, max_length: int = 256):
    device = next(model.parameters()).device
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


@st.cache_data(show_spinner=False)
def get_recipe_embeddings(combined_texts: list[str]):
    tokenizer, model = load_bert_model()
    return get_bert_embeddings(combined_texts, tokenizer, model)


@st.cache_data(show_spinner=False)
def get_recommendations(fav_dish: str, df: pd.DataFrame, num_recommendations: int = 12):
    if "combined" not in df.columns:
        df = df.copy()
        df["combined"] = preprocess_combined(df)

    combined_list = df["combined"].fillna("").tolist()

    # Cached embeddings for recipes
    recipe_embeddings = get_recipe_embeddings(combined_list)

    # Embedding for query
    tokenizer, model = load_bert_model()
    fav_emb = get_bert_embeddings([fav_dish], tokenizer, model)

    cosine_sim = cosine_similarity(fav_emb, recipe_embeddings)
    # Avoid returning empty / same-y results if dataset small
    k = min(num_recommendations, len(df))
    similar_indices = cosine_sim.argsort()[0, -k:][::-1]

    return df.iloc[similar_indices].reset_index(drop=True)


# ---------------------------
# Images / scraping
# ---------------------------
@st.cache_data(show_spinner=False)
def scrape_recipe_image(url: str):
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Your site-specific selector (guarded)
        div = soup.find("div", class_="recipe-image")
        img_tag = div.find("img") if div else None

        if img_tag and img_tag.get("src"):
            return urljoin(url, img_tag["src"])

        # Fallback: first reasonable <img>
        fallback = soup.find("img")
        if fallback and fallback.get("src"):
            return urljoin(url, fallback["src"])

    except Exception:
        return None

    return None


# ---------------------------
# Search / filters
# ---------------------------
@st.cache_data(show_spinner=False)
def search_recipes(df: pd.DataFrame, cuisine: str, course: str, diet: str, max_total_time: int):
    results = df[df["TotalTimeInMins"] <= int(max_total_time)]

    if cuisine != "Any":
        results = results[results["Cuisine"] == cuisine]
    if course != "Any":
        results = results[results["Course"] == course]
    if diet != "Any":
        results = results[results["Diet"] == diet]

    return results.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_applied_filters(cuisine, course, diet, max_total_time, default_max_time: int = 360):
    filters = []
    if cuisine != "Any":
        filters.append(f"Cuisine: {cuisine}")
    if course != "Any":
        filters.append(f"Course: {course}")
    if diet != "Any":
        filters.append(f"Diet: {diet}")
    if int(max_total_time) != int(default_max_time):
        filters.append(f"Max Time: {int(max_total_time)} mins")
    return filters


# ---------------------------
# UI helpers (not cached)
# ---------------------------
def show_recipe_details(recipe: pd.Series):
    st.subheader(str(recipe.get("RecipeName", "")))

    img_url = scrape_recipe_image(str(recipe.get("URL", "")))
    if img_url:
        st.image(img_url, width=300)

    st.write(f"**Ingredients:** {recipe.get('Ingredients', '')}")
    st.write(f"**Total Time:** {int(recipe.get('TotalTimeInMins', 0))} minutes")
    st.write(
        f"**Cuisine:** {recipe.get('Cuisine', '')}, "
        f"**Course:** {recipe.get('Course', '')}, "
        f"**Diet:** {recipe.get('Diet', '')}"
    )

    st.subheader("Instructions:")
    st.write(recipe.get("Instructions", ""))

    url = str(recipe.get("URL", "")).strip()
    if url:
        st.markdown(f"[View full recipe]({url})")


def scroll_to_recipe_details():
    components.html(
        """
        <script>
            const element = document.getElementById("recipe-details");
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        </script>
        """,
        height=0,
    )


def display_search_results(results: pd.DataFrame):
    # Session defaults
    if "show_recipe_details" not in st.session_state:
        st.session_state.show_recipe_details = None
    if "page" not in st.session_state:
        st.session_state.page = 1

    # Details panel
    if st.session_state.show_recipe_details is not None and len(results) > 0:
        idx = st.session_state.show_recipe_details
        idx = max(0, min(idx, len(results) - 1))
        recipe = results.iloc[idx]

        scroll_to_recipe_details()
        st.markdown('<a id="recipe-details"></a>', unsafe_allow_html=True)

        with st.expander("Recipe Details", expanded=True):
            show_recipe_details(recipe)
            if st.button("Close Recipe", key="close_recipe_btn"):
                st.session_state.show_recipe_details = None
                st.rerun()

    if len(results) == 0:
        st.write("No recipes found. Try adjusting your filters.")
        return

    # Pagination
    per_page = 6
    total_pages = (len(results) - 1) // per_page + 1
    st.session_state.page = max(1, min(st.session_state.page, total_pages))

    start_idx = (st.session_state.page - 1) * per_page
    end_idx = start_idx + per_page
    page_results = results.iloc[start_idx:end_idx]

    cols = st.columns(3)

    for local_idx, (_, row) in enumerate(page_results.iterrows()):
        global_idx = start_idx + local_idx
        with cols[local_idx % 3]:
            img = scrape_recipe_image(str(row.get("URL", "")))
            img_html = f'<img src="{img}" class="recipe-image">' if img else ""

            st.markdown(
                f"""
                <div class="recipe-card">
                    {img_html}
                    <div class="recipe-title">{row.get('RecipeName','')}</div>
                    <div class="recipe-info">
                        Cuisine: {row.get('Cuisine','')}<br>
                        Course: {row.get('Course','')}<br>
                        Total Time: {int(row.get('TotalTimeInMins',0))} minutes
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_key = f"show_{global_idx}_{row.get('URL','')}"
            if st.button("Show recipe", key=btn_key, use_container_width=True):
                st.session_state.show_recipe_details = global_idx
                st.rerun()

    st.markdown("---")

    nav1, nav2, nav3 = st.columns([2, 8, 1])

    with nav1:
        if st.session_state.page > 1:
            if st.button("< Previous", key="prev_page"):
                st.session_state.page -= 1
                st.session_state.show_recipe_details = None
                st.rerun()

    with nav2:
        st.markdown(
            f'<div style="text-align: center;">Page {st.session_state.page} of {total_pages}</div>',
            unsafe_allow_html=True,
        )

    with nav3:
        if st.session_state.page < total_pages:
            if st.button("Next >", key="next_page"):
                st.session_state.page += 1
                st.session_state.show_recipe_details = None
                st.rerun()


@st.cache_data(show_spinner=False)
def display_recommendations(query: str):
    df = load_data()
    return get_recommendations(query, df)