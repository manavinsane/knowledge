from fastapi import APIRouter

import logging


from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.hash import get_password_hash, verify_password
from src.core.jwt import (
    create_access_token_from_refresh_token,
    create_token_pair,
    decode_token,
)
from src.database import get_db
from src.model.user import User
from src.schema.user_schema import (
    AuthResponse,
    LoginRequest,
    RefreshTokenRequest,
    RefreshTokenResponse,
    RegisterRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register_user(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(select(User).where(User.email == payload.email))
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    hashed_password = get_password_hash(payload.password)

    new_user = User(
        username=payload.username,
        email=payload.email,
        password=hashed_password,
        role=payload.role,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    token_payload = {
        "id": str(new_user.id),
        "email": new_user.email,
    }

    access_token, refresh_token = create_token_pair(token_payload)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        username=new_user.username,
        email=new_user.email,
        role=new_user.role,
        created_at=new_user.created_at,
        updated_at=new_user.updated_at,
    )



@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
):

    stmt = select(User).where(User.email == payload.email)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(payload.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token_payload = {
        "id": str(user.id),
        "email": user.email,
    }

    access_token, refresh_token = create_token_pair(token_payload)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }

@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        decoded = decode_token(payload.refresh_token)

        if decoded is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
            )

        if decoded.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        user_id = decoded.get("id")

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        new_access_token = create_access_token_from_refresh_token(payload.refresh_token)

        if not new_access_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token expired",
            )

        return RefreshTokenResponse(access_token=new_access_token)

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
