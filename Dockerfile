FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Run CPU inference in the small Render/container environment. Install the CPU
# wheel first; pip then keeps this already-satisfied torch version when it
# resolves the remaining requirements.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501 8000
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
