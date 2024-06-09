from fastapi import FastAPI
from src.routes.api import asr

app = FastAPI()
app.include_router(asr.router)


@app.get("/")
def root():
    return {"res": "FastAPI is up and running!"}
