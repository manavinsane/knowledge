from pathlib import Path

from src.model.user import UserRole

PUBLIC_VISIBILITY = "PUBLIC"

ROLE_VISIBILITY = {
    UserRole.ADMIN: [PUBLIC_VISIBILITY, UserRole.ADMIN.value, UserRole.HR.value, UserRole.EMPLOYEE.value],
    UserRole.HR: [PUBLIC_VISIBILITY, UserRole.HR.value],
    UserRole.EMPLOYEE: [PUBLIC_VISIBILITY, UserRole.EMPLOYEE.value],
}


def visibility_from_source(source: str) -> str:
    parts = {part.lower() for part in Path(source).parts}

    if "admin" in parts:
        return UserRole.ADMIN.value
    if "hr" in parts:
        return UserRole.HR.value
    if "employee" in parts or "employees" in parts:
        return UserRole.EMPLOYEE.value

    return PUBLIC_VISIBILITY


def metadata_filter_for_role(role: UserRole | str) -> dict:
    user_role = role if isinstance(role, UserRole) else UserRole(role)
    return {"visibility": {"$in": ROLE_VISIBILITY[user_role]}}
