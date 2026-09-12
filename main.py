from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import process_data_insight
from core.actions.queryexecutor import QueryExecutor
from core.config import ALLOWED_ORIGINS, DATA_INSIGHT_ABOUT_CONFIGURATION
from core.connectors.LLMClient import LLMClient
from core.logger import get_logger, set_session_id

app = FastAPI()

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize core connections
llm_client = LLMClient()
query_executor = QueryExecutor()

logger = get_logger(__name__)

class QueryRequest(BaseModel):
    """
    QueryRequest defines the structure of the request payload for the /run_sql_agent endpoint.
    """
    session_id: str
    user_query: str
    chat_id: str

@app.post("/business-solutions/sql-agent/run")
async def run_sql_agent(request: QueryRequest):
    """
    Endpoint to process the SQL agent functionality.

    Args:
        request (QueryRequest): The request object containing user information and query details.

    Returns:
        dict: The processed result from the main data insight function.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        set_session_id(request.session_id)
        
        # Call the main processing function from app.py
        result = process_data_insight(
            session_id=request.session_id,
            user_query=request.user_query,
            chat_id=request.chat_id,
            llm_client=llm_client,
            query_executor=query_executor
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in run_sql_agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/business-solutions/sql-agent/health_check", status_code=status.HTTP_200_OK)
@app.get("/health_check", status_code=status.HTTP_200_OK)
async def root():
    """
    Health check endpoint to ensure the API service is running properly.

    Returns:
        dict: A dictionary with a welcome message to indicate the service is online.
    """
    return {"message": "Welcome to the SQL-Agent Service API"}


@app.get("/business-solutions/sql-agent/about")
async def about():
    """
    Provides information about the sql-agent service.

    Returns:
        dict: A dictionary containing information about the service.
    """
    return DATA_INSIGHT_ABOUT_CONFIGURATION


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)