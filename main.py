from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import json

from rag_pipeline1 import run_financial_rag_stream

app = FastAPI(
    title="Financial Intelligence API",
    version="1.0"
)

class FinancialRequest(BaseModel):
    query: str
    mode: str = "detailed"

@app.get("/")
def health():
    return {
        "status": "running"
    }

@app.post("/query")
def query(request: FinancialRequest):

    def generate_events():

        for event in run_financial_rag_stream(
            query=request.query,
            mode=request.mode
        ):
            yield (
                f"data: {json.dumps(event)}\n\n"
            )
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream"
    )