from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Enum as SQLAlchemyEnum
from sqlmodel import Field, SQLModel


class UserRole(str, Enum):
    HR = "HR"
    EMPLOYEE = "EMPLOYEE"
    ADMIN = "ADMIN"


class User(SQLModel, table=True):

    __tablename__ = "users"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        index=True,
        nullable=False
    )

    email: str = Field(
        unique=True,
        index=True,
        nullable=False,
        max_length=255
    )

    username: str = Field(
        unique=True,
        index=True,
        nullable=False,
        max_length=100
    )

    password: str = Field(
        nullable=False,
        max_length=255
    )

    is_active: bool = Field(
        default=True,
        nullable=False
    )

    role: UserRole = Field(
        default=UserRole.EMPLOYEE,
        sa_column=Column(
            SQLAlchemyEnum(
                UserRole,
                name="user_role",
                values_callable=lambda roles: [role.value for role in roles],
            ),
            nullable=False,
            server_default=UserRole.EMPLOYEE.value,
        ),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
