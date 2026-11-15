# THEIA web app.   docker build -t theia .   then   docker run -p 8501:8501 theia
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# code, trained model + processed open data, globe textures, fonts, logo
COPY assets ./assets
COPY static ./static
COPY .streamlit ./.streamlit
COPY streamlit_app.py ./
RUN useradd -m theia && chown -R theia /app
USER theia
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT','8501') + '/_stcore/health')"
# Most hosts (Render, Railway, Fly, Cloud Run) tell the app which port to use in $PORT.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
