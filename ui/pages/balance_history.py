import streamlit as st
import pandas as pd
from utils import check_auth, sign_in, get_me, get_balance_transactions

# Check authentication (if not authenticated - redirect to sign in page)
token = check_auth()
user_data = get_me(token)

st.title("История баланса")
st.write("##### Список всех транзакций")

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


col1, col2 = st.columns([0.15, 0.75])
with col1:
    if st.button("< Назад"):
        st.switch_page("main.py")
with col2:
    st.write("")

st.write("####")

balance_history = get_balance_transactions(user_data.get("id"), token)

if balance_history:
    h_1, h_2, h_3, h_4 = st.columns(4)
    h_1.write("**Дата**")
    h_2.write("**Сумма**")
    h_3.write("**Тип**")
    h_4.write("**Описание**")

    for tx in balance_history:
        col1, col2, col3, col4 = st.columns(4)

        date_str = tx.get("created_at")[:16].replace("T", " ")

        col1.write(date_str)
        col2.write(tx.get("transaction_amount"))
        col3.write(tx.get("transaction_type"))
        col4.write(tx.get("description"))
else:
    st.info("Транзакции не найдены")

