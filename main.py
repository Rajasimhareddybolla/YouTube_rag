import streamlit as st
import textwrap
import helper
db_s = {}
def main():
    st.sidebar.title("Search")
    url = st.sidebar.text_input("URL")
    query = st.sidebar.text_input("Query")
    submit_button = st.sidebar.button("Submit")

    if submit_button:
        # Perform the search using serp api
        if url not in db_s:
            db = helper.youtube_url_to_db(url)
            db_s[url] = db

        else:
            st.write("used prev_db")
            db = db_s[url]
        summery=  helper.get_resp_query(db , query , k = 3)
        
        st.write("Summary:", textwrap.fill(summery, width=85))

if __name__ == '__main__':
    main()
