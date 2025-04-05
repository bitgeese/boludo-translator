"""
Main FastAPI application that mounts the Chainlit app.
This provides a proper deployment architecture for Render.com
"""

from chainlit.utils import mount_chainlit
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address


# Custom key function that exempts the health endpoint from rate limiting
def get_key_func(request: Request):
    # If this is a request to the health endpoint,
    # generate a unique key to disabling rate limiting
    if request.url.path == "/health":
        # Return a unique key for each request to bypass rate limiting
        import time

        return f"health-{time.time()}"

    # For all other requests, use the client's IP address
    return get_remote_address(request)


# Define the rate limiter with our custom key function
limiter = Limiter(key_func=get_key_func, default_limits=["5/minute"])

# Create FastAPI app
app = FastAPI(
    title="Argentine Spanish Translator",
    description="A translator that specializes in Argentine Spanish expressions "
    "and slang",
    version="1.0.0",
)

# Add the limiter to the app state
app.state.limiter = limiter

# Add the middleware to handle rate limiting
app.add_middleware(SlowAPIMiddleware)

# Add exception handler for rate limit exceeded
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Health check endpoint for Render
@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "ok"}


# Mount Chainlit to the FastAPI app
# The limiter middleware will apply to this mounted app as well
mount_chainlit(app=app, target="app.py", path="/")

# If running this script directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
