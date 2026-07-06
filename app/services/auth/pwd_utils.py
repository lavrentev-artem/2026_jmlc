import bcrypt


def hash_password(plain_password: str) -> str:
    """
    Creates a password hash
    Args:
        plain_password: a plain password sent by the user
    Returns:
        hashed_password: hash from the plain_password
    """
    try:
        password_bytes = plain_password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
        return hashed_password

    except Exception as e:
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    Args:
        plain_password: a plain password to be verified
        hashed_password: hashed password to be verified against
    Returns:
        If the password matches the hashed_password, return True
    """
    password_bytes = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)
