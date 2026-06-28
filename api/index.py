# Vercel serverless entrypoint (and local uvicorn target): builds the FastAPI app.
# `app` is assigned from a call so Vercel's FastAPI detector recognizes it here.
from backend.main import create_app

app = create_app()
