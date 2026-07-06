import streamlit as st
import requests
import os
from decimal import Decimal


API_URL = os.getenv("API_URL", "http://app:8080")

def sign_up(full_name: str, email: str, password: str):
    """
    Registers a new user and logs him in instantly
    Args:
        full_name: User's full name
        email: User's email
        password: User's password
    Returns:
        Access token
    """
    payload = {
        "full_name": full_name,
        "email": email,
        "password": password,
    }
    response = requests.post(
        url=f"{API_URL}/auth/sign_up",
        json=payload,
    )

    if response.status_code == 200:
        return response.json()
    else:
        return None


def sign_in(email: str, password: str):
    """
    Signs in a user and returns access token
    Args:
        email: User's email
        password: User's password
    Returns:
        Access token
    """
    payload = {
        "username": email,
        "password": password,
    }
    response = requests.post(
        url=f"{API_URL}/auth/sign_in",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code == 200:
        return response.json()
    else:
        return None


def sign_out():
    st.session_state.access_token = None
    st.session_state.user_data = None
    st.switch_page("pages/sign_in.py")


def check_auth():
    """
    Checks if user is logged in
    Returns:
        Access token
    """
    if not st.session_state.get("access_token"):
        st.switch_page("pages/sign_in.py")
        st.stop()
    return st.session_state.access_token


def get_me(token: str):
    """
    Retrieves user's info by access token
    Args:
        token: Access token
    Returns:
        User info
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/user/me", headers=headers)
    return response.json()


def get_user_by_email(email: str, token: str):
    """Get user by email"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        url=f"{API_URL}/user/get_by_email",
        params={"email": email},
        headers=headers
    )
    response.raise_for_status()
    return response.json()


def get_balance(user_id: str, token: str):
    """Get user's current balance amount"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/balance/{user_id}/amount", headers=headers)
    response.raise_for_status()
    return response.json()


def get_balance_transactions(user_id: str, token: str):
    """Get user's balance transactions"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/balance/{user_id}/transactions", headers=headers)
    response.raise_for_status()
    return response.json()


def get_operation_list(user_id: str, token: str):
    """Get user's operations"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        url=f"{API_URL}/operation/list",
        params={"user_id": user_id},
        headers=headers)
    response.raise_for_status()
    return response.json()


def create_prompt(user_id: str, operation_input: str, token: str):
    """Prompt ML Service"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        url=f"{API_URL}/operation/prompt",
        headers = headers,
        json={
            "user_id": user_id,
            "operation_input": operation_input,
        }
    )
    response.raise_for_status()
    return response.json()


def topup_balance(user_id: str, transaction_amount: Decimal, token: str):
    """Prompt ML Service"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        url=f"{API_URL}/balance/{user_id}/topup",
        headers = headers,
        json={
            "transaction_amount": str(transaction_amount),
            "description": "Topup from Web-UI",
        }
    )
    response.raise_for_status()
    return response.json()

