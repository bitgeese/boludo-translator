"""
Main FastAPI application that mounts the Chainlit app.
This provides a proper deployment architecture for Render.com
"""

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

# Create FastAPI app
app = FastAPI(
    title="Argentine Spanish Translator",
    description="A translator that specializes in Argentine Spanish expressions "
    "and slang",
    version="1.0.0",
)


# Health check endpoint for Render
@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "ok"}


# Mount Chainlit to the FastAPI app
mount_chainlit(app=app, target="app.py", path="/")

# If running this script directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
