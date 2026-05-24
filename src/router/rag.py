from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.auth import get_current_user
from src.model.user import User, UserRole
from src.rag.chatbot import answer_question

router = APIRouter(prefix="/v1/rag", tags=["rag"])
legacy_router = APIRouter(tags=["rag"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    role: UserRole


@router.post("/ask", response_model=AskResponse)
@legacy_router.post("/ask", response_model=AskResponse)
async def ask_question(
    payload: AskRequest,
    current_user: User = Depends(get_current_user),
):
    answer = answer_question(payload.question, role=current_user.role)
    return AskResponse(
        question=payload.question,
        answer=answer,
        role=current_user.role,
    )
