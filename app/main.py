import streamlit as st
import os
import auth_functions
from functions import load_data, get_css_path, display_recommendations, search_recipes, display_search_results

st.set_page_config(layout="wide")
st.markdown(f'<div style="text-align: center;"><h1>🍳 AI Recipe Recommender</h1></div>', unsafe_allow_html=True)
st.write("---")

if 'user_info' not in st.session_state:
    col1, col2, col3 = st.columns([1, 2, 1])
    do_you_have_an_account = col2.selectbox(label='Do you have an account?', options=('Yes', 'No', 'I forgot my password'))
    auth_form = col2.form(key='Authentication form', clear_on_submit=False)
    email = auth_form.text_input(label='Email')
    password = auth_form.text_input(label='Password', type='password') if do_you_have_an_account in {'Yes', 'No'} else auth_form.empty()
    auth_notification = col2.empty()

    if do_you_have_an_account == 'Yes' and auth_form.form_submit_button(label='Sign In'):
        with auth_notification, st.spinner('Signing in'):
            auth_functions.sign_in(email, password)

    elif do_you_have_an_account == 'No' and auth_form.form_submit_button(label='Create Account'):
        with auth_notification, st.spinner('Creating account'):
            auth_functions.create_account(email, password)

    elif do_you_have_an_account == 'I forgot my password' and auth_form.form_submit_button(label='Send Password Reset Email'):
        with auth_notification, st.spinner('Sending password reset link'):
            auth_functions.reset_password(email)

    if 'auth_success' in st.session_state:
        auth_notification.success(st.session_state.auth_success)
        del st.session_state.auth_success
    elif 'auth_warning' in st.session_state:
        auth_notification.warning(st.session_state.auth_warning)
        del st.session_state.auth_warning

else:
    def main():
        col1, col2, col3 = st.columns([2, 8, 2])
        with col2:
            email = st.session_state.user_info.get("email")
            st.markdown(f"<h3>Welcome, {email}</h3>", unsafe_allow_html=True)
        with col3:
            st.button(label='Sign Out', on_click=auth_functions.sign_out)
        st.write("---")

        css_path = get_css_path()
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'food.csv')
        df = load_data(csv_path)
        fav_dish = st.text_input("What's on your mind")

        if st.button("Get Recommendations"):
            recommendation = display_recommendations(df, fav_dish)
            st.session_state['recommendation'] = recommendation
        if 'recommendation' in st.session_state:
            recommendation = display_recommendations(df, fav_dish)
            st.sidebar.header("Search Options")
            recipe_name = st.sidebar.text_input("Recipe name")
            ingredient = st.sidebar.text_input("Main ingredient")
            cuisine = st.sidebar.selectbox("Cuisine", ["Any"] + sorted(recommendation['Cuisine'].unique().tolist()))
            course = st.sidebar.selectbox("Course", ["Any"] + sorted(recommendation['Course'].unique().tolist()))
            diet = st.sidebar.selectbox("Diet", ["Any"] + sorted(recommendation['Diet'].unique().tolist()))
            max_total_time = st.sidebar.slider("Maximum time (minutes):", 0, 360, 60)
            if st.sidebar.button("Search Recipes in the recommendation"):
                results = search_recipes(recommendation, recipe_name, ingredient, cuisine, course, diet, max_total_time)
                display_search_results(results)
            else:
                display_search_results(recommendation)

    if __name__ == "__main__":
        main()
