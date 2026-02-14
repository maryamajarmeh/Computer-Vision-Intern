
FROM python:3.10-slim



ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY teeth_model.h5 .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
