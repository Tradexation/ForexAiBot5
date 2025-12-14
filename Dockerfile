# Dockerfile

# Use an official Python runtime as a base image
FROM python:3.11-slim 

# Set the working directory
WORKDIR /app 

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the application code (including main.py)
COPY . /app

# The default command to run the application using Gunicorn
# This loads the Flask app 'app' from the file 'main.py'

CMD gunicorn main:app --bind 0.0.0.0:$PORT
