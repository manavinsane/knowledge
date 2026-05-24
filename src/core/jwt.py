from datetime import datetime, timedelta, timezone

from jose import jwt

import src.core.config as config


def create_token_pair(payload: dict):
    """
    Create access and refresh token pair with different expiration times

    Args:
        payload: Data to encode in the token (usually user info)
        access_expire_minutes: Access token expiration time in minutes (default: 30)
        refresh_expire_days: Refresh token expiration time in days (default: 7)
    """
    access_payload = payload.copy()
    access_exp = datetime.now(timezone.utc) + timedelta(
        minutes=config.ACCESS_TOKEN_EXPIRES_MINUTES
    )
    access_payload.update({"exp": access_exp, "type": "access"})
    access_token = jwt.encode(
        access_payload, config.SECRET_KEY, algorithm=config.ALGORITHM
    )

    resfresh_payload = payload.copy()
    refresh_exp = datetime.now(timezone.utc) + timedelta(
        minutes=config.REFRESH_TOKEN_EXPIRES_MINUTES
    )
    resfresh_payload.update({"exp": refresh_exp, "type": "refresh"})
    refresh_token = jwt.encode(
        resfresh_payload, config.SECRET_KEY, algorithm=config.ALGORITHM
    )

    return access_token, refresh_token


def decode_token(token: str):
    """
    Decode and validate JWT token

    Returns:
        dict: Decoded payload if valid
        None: If token is expired or invalid
    """
    try:
        decoded_token = jwt.decode(
            token, config.SECRET_KEY, algorithms=[config.ALGORITHM]
        )
        return decoded_token
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def create_access_token_from_refresh_token(
    refresh_token: str, access_expire_minutes: int = 30
):
    """
    Create new access token from valid refresh token

    Args:
        refresh_token: Valid refresh token
        access_expire_minutes: New access token expiration time in minutes

    Returns:
        str: New access token if refresh token is valid
        None: If refresh token is expired or invalid
    """
    try:
        payload = jwt.decode(
            refresh_token, config.SECRET_KEY, algorithms=[config.ALGORITHM]
        )

        if payload.get("type") != "refresh":
            return None

        # Create new access token
        access_payload = {k: v for k, v in payload.items() if k not in ["exp", "type"]}
        access_expire = datetime.now(timezone.utc) + timedelta(
            minutes=access_expire_minutes
        )
        access_payload.update({"exp": access_expire, "type": "access"})

        access_token = jwt.encode(
            access_payload, config.SECRET_KEY, algorithm=config.ALGORITHM
        )
        return access_token

    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
