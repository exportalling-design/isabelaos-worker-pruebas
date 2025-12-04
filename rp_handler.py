# rp_handler.py - Worker Serverless de IsabelaOS Studio en RunPod
import os
import io
import base64
from pathlib import Path
from typing import Any, Dict

from PIL import Image
import runpod

# ------------------ Configuración de modelo ------------------

# ID del modelo en Hugging Face (puedes cambiarlo si usas otro)
HF_MODEL_ID = os.getenv(
    "ISE_BASE_MODEL_ID",
    "SG161222/Realistic_Vision_V5.1_noVAE"  # ejemplo típico de RealisticVision
)

# Carpeta donde RunPod cachea el modelo
MODELS_DIR = Path("/runpod/volumes/isabelaos/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta para guardar imágenes
IMAGES_DIR = Path("/runpod/volumes/isabelaos/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta para guardar videos (GIF por ahora)
VIDEOS_DIR = Path("/runpod/volumes/isabelaos/videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline global (se inicializa una sola vez)
_PIPELINE = None


# ------------------ Utilidades ------------------


def _ensure_hf_cached_download_alias():
    """
    Parche para versiones nuevas de huggingface_hub que quitaron cached_download.
    Diffusers todavía la llama en algunos puntos.
    """
    try:
        import huggingface_hub as h
        if not hasattr(h, "cached_download"):
            from huggingface_hub import hf_hub_download as _hfd
            setattr(h, "cached_download", _hfd)
    except Exception:
        pass


def _b64_from_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_from_file(path: Path) -> str:
    """Convierte cualquier archivo binario (GIF, MP4, etc.) a base64."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ok(**kwargs) -> Dict[str, Any]:
    out = {"ok": True}
    out.update(kwargs)
    return out


def _err(msg: str, **kwargs) -> Dict[str, Any]:
    out = {"ok": False, "error": msg}
    out.update(kwargs)
    return out


# ------------------ Inicialización del modelo ------------------


def _get_pipeline():
    """
    Crea o devuelve el pipeline global de Stable Diffusion.
    Usa HF_MODEL_ID y HF_TOKEN (si está definido).
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _ensure_hf_cached_download_alias()

    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Opcional: login a Hugging Face si tienes token
    hf_token = os.getenv("HF_TOKEN", None)
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            # si falla el login, igual intentamos descargar
            pass

    print(f"[ISE] Cargando modelo {HF_MODEL_ID} en {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
        safety_checker=None
    )
    pipe = pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _PIPELINE = pipe
    print("[ISE] Modelo cargado.")
    return _PIPELINE


# ------------------ Lógica principal de generación (IMAGEN) ------------------


def generate_image_from_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    input_data viene de event["input"] en RunPod.
    Espera:
      - prompt (str)
      - negative_prompt (str, opcional)
      - width, height (int, opcionales)
      - steps (int, opcional)
      - guidance_scale (float, opcional)
      - seed (int, opcional)
    """
    prompt = input_data.get(
        "prompt",
        "a beautiful cinematic portrait, high detail, dramatic lighting"
    )
    negative = input_data.get(
        "negative_prompt",
        "low quality, blurry, deformed, text"
    )

    width = int(input_data.get("width", 512))
    height = int(input_data.get("height", 512))
    steps = int(input_data.get("steps", 22))
    guidance = float(input_data.get("guidance_scale", 7.5))

    seed = input_data.get("seed", None)

    pipe = _get_pipeline()

    import torch
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=pipe.device.type).manual_seed(int(seed))
        except Exception:
            generator = None

    print(f"[ISE] Generando imagen: prompt='{prompt[:60]}...' "
          f"({width}x{height}, steps={steps}, guidance={guidance})")

    result = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        negative_prompt=negative,
        generator=generator
    )

    img = result.images[0]

    # Guardar imagen en el volumen
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    name = f"isabela_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = IMAGES_DIR / name
    img.save(out_path, format="PNG")

    return _ok(
        image_path=str(out_path),
        image_b64=_b64_from_pil(img)
    )


# ------------------ Generación de VIDEO básico (GIF) ------------------


def generate_video_from_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un "video" básico como GIF a partir de varios frames
    generados con la misma pipeline de imagen.

    Campos esperados en input_data:
      - prompt (str)
      - negative_prompt (str, opcional)
      - width, height (int, opcionales)
      - steps (int, opcional)
      - num_frames (int, opcional, defecto 48)
      - fps (int, opcional, defecto 8)  -> 48 frames / 8 fps = 6 segundos
      - seed (int, opcional)
    """
    prompt = input_data.get(
        "prompt",
        "cinematic shot, high detail, 4k, ultra realistic"
    )
    negative = input_data.get(
        "negative_prompt",
        "low quality, blurry, deformed, text"
    )

    width = int(input_data.get("width", 512))
    height = int(input_data.get("height", 512))
    steps = int(input_data.get("steps", 20))

    # 48 frames a 8 fps ≈ 6 segundos
    num_frames = int(input_data.get("num_frames", 48))
    fps = int(input_data.get("fps", 8))

    seed = input_data.get("seed", None)

    pipe = _get_pipeline()

    import torch

    if seed is None:
        # si no viene seed, escogemos una al azar y luego sumamos sobre ella
        seed_base = torch.randint(0, 2**31 - 1, (1,)).item()
    else:
        seed_base = int(seed)

    print(
        f"[ISE] Generando VIDEO básico: prompt='{prompt[:60]}...' "
        f"({width}x{height}, steps={steps}, frames={num_frames}, fps={fps})"
    )

    frames = []

    for i in range(num_frames):
        frame_seed = seed_base + i
        generator = torch.Generator(device=pipe.device.type).manual_seed(frame_seed)

        result = pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=7.5,
            negative_prompt=negative,
            generator=generator,
        )

        img = result.images[0]
        frames.append(img)

    # Guardar GIF en el volumen
    from datetime import datetime

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"isabela_video_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.gif"
    out_path = VIDEOS_DIR / name

    # duración de cada frame en ms (1000 ms / fps)
    duration_ms = int(1000 / max(fps, 1))

    # guardamos como GIF animado
    first_frame, *rest = frames
    first_frame.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        format="GIF",
    )

    print(f"[ISE] VIDEO básico guardado en: {out_path}")

    video_b64 = _b64_from_file(out_path)

    return _ok(
        video_path=str(out_path),
        video_b64=video_b64,
        format="gif",
        fps=fps,
        num_frames=num_frames,
    )


# ------------------ Handler de RunPod ------------------


def handler(event):
    """
    Formato típico:
    {
      "input": {
        "action": "image" | "video" | "health",
        "prompt": "...",
        "negative_prompt": "...",
        "width": 512,
        "height": 512,
        "steps": 22
      }
    }
    """
    try:
        data = (event or {}).get("input") or {}
        action = str(data.get("action", "image")).lower()

        # Healthcheck
        if action == "health":
            return _ok(status="up", model_id=HF_MODEL_ID)

        # Imagen estática (lo que ya funcionaba)
        if action == "image":
            return generate_image_from_input(data)

        # Video básico (GIF desde varios frames)
        if action == "video":
            return generate_video_from_input(data)

        # Aquí después podemos ir agregando:
        # if action == "cinecam": ...
        # if action == "bodysync": ...
        # if action == "script2film": ...

        return _err(f"Unknown action '{action}'")

    except Exception as e:
        return _err(f"Exception: {e!r}")


runpod.serverless.start({"handler": handler})

