import requests
import streamlit as st
from streamlit import rerun
from datetime import datetime
from decimal import Decimal
import traceback

from streamlit.runtime.scriptrunner_utils.exceptions import StopException, RerunException

from utils import (check_auth, get_me, get_operation_list,
                   get_balance, create_prompt, topup_balance, sign_out)

# noinspection PyBroadException
try:
    st.set_page_config(page_title="Чатбот", layout="centered")

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

    #-- Authentication
    # Check access token
    # if "access_token" not in st.session_state:
    #     st.session_state.access_token = None
    # if "user_data" not in st.session_state:
    #     st.session_state.user_data = None

    # Check authentication (if not authenticated - redirect to sign in page)
    token = check_auth()
    user_data = get_me(token)

    # Render header
    st.title("Чатбот")


    if user_data:
        user_id = user_data.get("id")
        full_name = user_data.get("full_name")

        # Render header
        # st.subheader(f'Welcome, {full_name}!')

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader(f'Привет, {full_name}!')
        with col2:
            st.write('')
            if st.button(label="Выйти", type="secondary"):
                sign_out()

        st.write('###')


        # Render balance widget
        col_balance, col_topup, col_user_info = st.columns([0.25, 0.25, 0.5])

        # Get balance
        balance_amount = get_balance(user_id, token)

        with col_balance:
            st.metric(label='Текущий баланс', value=f'{balance_amount.get("current_amount")} cr.')

        with col_topup:
            # st.write("")
            # st.write("")
            topup_popover = st.popover('➕ Top-up')
            st.write("")
            if st.button("История баланса", key="btn_balance_history"):
                st.switch_page("pages/balance_history.py")

            with topup_popover:
                st.write('Enter amount of top-up in credits:')
                topup_amount = st.number_input(
                    label='Top-up amount',
                    min_value=1.00,
                    max_value=1000.00,
                    value=10.00,
                    step=1.00,
                    format="%.2f",
                    label_visibility="collapsed"
                )
                if st.button(label='Top-up', use_container_width=True):
                    topup_amount_decimal = Decimal(str(topup_amount))

                    with st.spinner('Processing top-up...'):
                        if topup_balance(user_id, topup_amount_decimal, token):
                            st.success(f'Top-up proceed successfully!')
                            st.rerun()
                        else:
                            st.error('Top-up failed')

        with col_user_info:
            st.write(f'**Email:** {user_data.get("email")}')
            st.write(f'**ID:** `{user_data.get("id")}`')

        st.divider()

        st.write("О чём ты хочешь поговорить?")

        # Scrollable area
        # Prompts history
        chat_container = st.container(height=500)

        with chat_container:
            operations = get_operation_list(user_id, token)

            if not operations:
                st.info('В диалоге пока что нет сообщений')
            else:
                # Render last 5 prompts
                for op in operations:
                    with st.chat_message('user'):
                        st.write(op.get('operation_input'))

                    with st.chat_message('assistant'):
                        st.write(op.get('operation_output'))


        #-- Prompt input field
        if operation_input := st.chat_input('Введите сообщение...'):
            with chat_container:
                with st.chat_message('user'):
                    st.write(operation_input)

            with st.spinner('Analyzing...'):
                try:
                    result = create_prompt(user_id, operation_input, token)
                    if result.get("status") == "failed":
                        st.error(result.get("error_message"))
                    st.rerun()
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 402:
                        st.error(e.response.json().get('message'))

    else:
        st.error(f'User not found')

except (StopException, RerunException):
    raise
except requests.exceptions.RequestException as e:
    st.error("Cannot connect to server")
except Exception as e:
    print(f"DEBUG ERROR: {e}")
    traceback.print_exc()
    st.error('Something went wrong')