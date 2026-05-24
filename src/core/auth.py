from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.jwt import decode_token
from src.database import get_db
from src.model.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login", auto_error=False)


async def get_current_user(
    bearer_token: str | None = Depends(oauth2_scheme),
    x_auth_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = x_auth_token or bearer_token
    if token is None:
        raise credentials_exception

    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    decoded = decode_token(token)
    if decoded is None or decoded.get("type") != "access":
        raise credentials_exception

    user_id = decoded.get("id")
    if not user_id:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception

    return user
