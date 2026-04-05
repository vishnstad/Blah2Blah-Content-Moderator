FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the content
COPY content_moderation_env/ /app/content_moderation_env/
COPY data/ /app/data/
COPY app.py /app/app.py
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml

# Set permissions for HF Spaces User
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR /app

EXPOSE 7860

CMD ["python", "app.py"]
