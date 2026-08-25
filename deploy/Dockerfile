# Only needed if you host somewhere other than Hugging Face Spaces.
# Spaces builds Gradio apps from requirements.txt on its own.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_bundle.joblib data.py app.py ./

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
CMD ["python", "app.py"]
