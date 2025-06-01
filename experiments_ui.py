import streamlit as st


def main():
    st.set_page_config(
        layout="wide",
        page_icon=":robot:",
        page_title="TST Lab",
        initial_sidebar_state="collapsed",
    )
    pg = st.navigation(
        [
            st.Page("experiments_lab.py", title="Lab"),
            st.Page("experiments_explorer.py", title="Experiments Explorer"),
        ],
        expanded=False,
    )
    pg.run()


if __name__ == "__main__":
    main()
