# เปลี่ยนจาก 3.11 เป็น 3.10 เพื่อให้รองรับ firthlogist
FROM python:3.10-slim

WORKDIR /app

# ติดตั้ง Git และเครื่องมือสำหรับ Build
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
