import streamlit as st
st.set_page_config(layout="wide")
import os
import auth_functions
from functions import load_data, get_css_path
from functions import display_recommendations, search_recipes
from functions import display_search_results,get_applied_filters
import streamlit.components.v1 as components

css_path = get_css_path()
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown(f'''<div id="heading" class="top-heading"><h2>🍳 AI Recipe Recommender</h2></div>''', unsafe_allow_html=True)
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
        col1, col2, col3 = st.columns([2,10,2])
        with col2:
            email = st.session_state.user_info.get("email")
            st.markdown(f'''<div  class="welcome-text"><h5>Welcome  {email}</h5></div>''', unsafe_allow_html=True)
        with col3:
            with st.expander('Profile'):
                st.button(label='Sign Out', on_click=auth_functions.sign_out)
        st.markdown("---")
        #csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'food.csv')   
        
        st.markdown("#### Please enter Ingredients/ recipe on your mind?")
        query = st.text_input("")
        if st.button("Get Recommendations") and query:
            recommendation = display_recommendations(query)
            st.session_state['recommendation'] = recommendation
            
        if 'recommendation' in st.session_state and st.session_state['recommendation'] is not None:
            recommendation = display_recommendations(query)
            st.sidebar.header("Filters")
            cuisine = st.sidebar.selectbox("Cuisine", ["Any"] + sorted(recommendation['Cuisine'].unique().tolist()))
            course = st.sidebar.selectbox("Course", ["Any"] + sorted(recommendation['Course'].unique().tolist()))
            diet = st.sidebar.selectbox("Diet", ["Any"] + sorted(recommendation['Diet'].unique().tolist()))
            max_total_time = float(st.sidebar.slider("Maximum time (minutes):", 0, 360, 360))
            placeholder = st.empty()
            
            if st.sidebar.button("Filter"): 
                if cuisine!="Any" or course!='Any' or diet!='Any' or max_total_time!='360':
                    results = search_recipes(recommendation,cuisine, course, diet, max_total_time)
                    applied_filters = get_applied_filters(cuisine, course, diet, max_total_time)
                    placeholder.write(f"Filters applied : {applied_filters}")
                    display_search_results(results)
                    
                else:
                    placeholder.empty()
                    display_search_results(recommendation)
                    
            else:
                display_search_results(recommendation)
        with st.expander('Delete Account'):
            password = st.text_input(label='Confirm your password', type='password')
            st.button(label='Delete', on_click=auth_functions.delete_account, args=[password], type='primary',key="delete_button")
        st.write("")

    if __name__ == "__main__":
        main()
