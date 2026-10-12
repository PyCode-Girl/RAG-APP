FROM python:3.12-slim

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user/app

WORKDIR /home/user/app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user src ./src

USER user
ENV PORT=7860
CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]