from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from services.rag_service import answer_question
from logger import logger

router = APIRouter()


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        result = answer_question(question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )