FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        libasound2-dev \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy<2" \
        six \
        pyaudio \
        edge_impulse_linux==1.2.2

CMD ["bash"]