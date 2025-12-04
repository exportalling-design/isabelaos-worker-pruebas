FROM runpod/serverless:3.10-onnx

WORKDIR /workspace

COPY . /workspace

RUN pip install --no-cache-dir -r requirements.txt

=======
# Imagen pública de Python con Debian slim
FROM python:3.10-slim

# Carpeta de trabajo dentro del contenedor
WORKDIR /workspace

# ---- Dependencias del sistema (mínimas) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Copiar requirements y instalarlos ----
COPY requirements.txt /workspace/requirements.txt

# Actualizar pip e instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Instalar PyTorch con CUDA 12.1 (GPU) ----
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Copiar el resto del código del worker ----
COPY . /workspace

# Puerto HTTP opcional (por si lo usas)
EXPOSE 8000

# Comando de inicio del worker RunPod
>>>>>>> 02791861f45a6ec242a8ef28e4251445bafee31f
CMD ["python", "-u", "rp_handler.py"]
