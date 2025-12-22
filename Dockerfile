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

# --- ส่วนที่เพิ่มเข้ามา ---
# 1. สร้าง User ใหม่ชื่อ streamlit (เพื่อความปลอดภัย ไม่ใช้ root)
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

# 2. สลับไปใช้ User นี้
USER streamlit

# 3. เพิ่ม Healthcheck (เพื่อให้รู้สถานะโปรแกรม)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/_stcore/health || exit 1
# ---------------------

EXPOSE 7860
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
