import streamlit as st
from utils import sign_up, sign_in, get_me

st.title("Регистрация")

# Switch off technical menu
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stStatusWidget"] {display: none;}
        .stDeployButton {display: none;}

        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem:
        }

        [data-testid="stAppHeader"] {
            height: 0px;
            display: none;
        }            
    </style>
""", unsafe_allow_html=True)

with st.form("sign_up_form"):
    full_name = st.text_input("Имя")
    email = st.text_input("Email")
    password = st.text_input(label="Пароль", type="password")
    repeated_password = st.text_input(label="Повторите пароль", type="password")
    submit = st.form_submit_button(label="Создать аккаунт")

    if submit:
        if password != repeated_password:
            st.error("Пароли не совпадают!")
        else:
            if user_data := sign_up(full_name, email, password):
                if token_data := sign_in(email, password):
                    st.session_state.access_token = token_data["access_token"]
                    st.session_state.user_data = get_me(token_data["access_token"])
                    st.success("Вы вошли!")
                    st.switch_page("main.py")
                else:
                    st.error("Неправильный Email или пароль")
            else:
                st.error("Ошибка создания аккаунта")

st.write('#####')
st.divider()

col1, col2 = st.columns([0.75, 0.15])
with col1:
    st.subheader("Уже зарегистрированы?")
with col2:
    st.write('')
    if st.button("Войти"):
        st.switch_page("pages/sign_in.py")