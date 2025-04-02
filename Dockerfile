FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Chainlit application
# Assumes your main Chainlit script is app.py
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"] 