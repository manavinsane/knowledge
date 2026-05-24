from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from src.model.user import UserRole


# ------------------------
# REQUEST SCHEMA
# ------------------------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    role: UserRole = UserRole.EMPLOYEE


# ------------------------
# USER RESPONSE SCHEMA
# ------------------------
class UserResponse(BaseModel):
    username: str
    email: EmailStr
    role: UserRole
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Important for SQLAlchemy


# ------------------------
# TOKEN RESPONSE
# ------------------------
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=72)


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    updated_at: datetime


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    access_token: str


# ------------------------
# PASSWORD RESET SCHEMAS
# ------------------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class VerifyOTPResponse(BaseModel):
    verified: bool
    message: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str = Field(..., min_length=8, max_length=72)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=72)
    new_password: str = Field(..., min_length=8, max_length=72)
