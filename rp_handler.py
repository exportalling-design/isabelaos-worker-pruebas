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

# ID del Motion Adapter para AnimateDiff (puedes sobreescribir por env)
ANIMATEDIFF_ADAPTER_ID = os.getenv(
    "ISE_ANIMATEDIFF_ADAPTER_ID",
    "guoyww/animatediff-motion-adapter-v1-5-2"  # ejemplo común, cámbialo si usas otro
)

# Carpeta donde RunPod cachea el modelo
MODELS_DIR = Path("/runpod/volumes/isabelaos/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta para guardar imágenes
IMAGES_DIR = Path("/runpod/volumes/isabelaos/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
# Carpeta para guardar videos
VIDEOS_DIR = Path("/runpod/volumes/isabelaos/videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Pipelines globales (se inicializan una sola vez)
_PIPELINE = None          # SD imagen
_AD_PIPELINE = None       # AnimateDiff
=======
# Carpeta para guardar videos (GIF por ahora)
VIDEOS_DIR = Path("/runpod/volumes/isabelaos/videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline global (se inicializa una sola vez)
_PIPELINE = None
>>>>>>> 1e21153fc489de1f976887f457fa090274d29dc3


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


<<<<<<< HEAD
def _hf_login_if_needed():
    """Login opcional a Hugging Face usando HF_TOKEN, reutilizable."""
    hf_token = os.getenv("HF_TOKEN", None)
    if not hf_token:
        return
    try:
        from huggingface_hub import login
        login(token=hf_token)
    except Exception:
        # Si falla el login, igual intentamos descargar
        pass


# ------------------ Inicialización del modelo (IMAGEN) ------------------
=======
# ------------------ Inicialización del modelo ------------------
>>>>>>> 1e21153fc489de1f976887f457fa090274d29dc3


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

    _hf_login_if_needed()

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


<<<<<<< HEAD
# ------------------ Inicialización AnimateDiff (VIDEO) ------------------


def _get_animatediff_pipeline():
    """
    Crea o devuelve el pipeline global de AnimateDiff.
    Usa el mismo modelo base HF_MODEL_ID + Motion Adapter.
    """
    global _AD_PIPELINE
    if _AD_PIPELINE is not None:
        return _AD_PIPELINE

    _ensure_hf_cached_download_alias()

    import torch
    from diffusers import StableDiffusionPipeline, MotionAdapter, AnimateDiffPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _hf_login_if_needed()

    print(f"[ISE] Cargando Motion Adapter AnimateDiff: {ANIMATEDIFF_ADAPTER_ID}")
    adapter = MotionAdapter.from_pretrained(
        ANIMATEDIFF_ADAPTER_ID,
        torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )

    print(f"[ISE] Cargando modelo base para AnimateDiff: {HF_MODEL_ID}")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
        safety_checker=None,
    )

    pipe = AnimateDiffPipeline.from_pipe(base_pipe, adapter)
    pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Algunos trucos de memoria (opcionales)
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    _AD_PIPELINE = pipe
    print("[ISE] Pipeline AnimateDiff cargado.")
    return _AD_PIPELINE


=======
>>>>>>> 1e21153fc489de1f976887f457fa090274d29dc3
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


<<<<<<< HEAD
# ------------------ Generación de VIDEO con AnimateDiff ------------------


def generate_animatediff_video_from_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un video usando AnimateDiff.

    Campos esperados en input_data:
      - prompt (str)
      - negative_prompt (str, opcional)
      - width, height (int, opcionales)
      - steps (int, opcional, defecto 16-24)
      - num_frames (int, opcional, defecto 16 o 24)
      - fps (int, opcional, defecto 8-12)
      - guidance_scale (float, opcional)
      - seed (int, opcional)
    """
    prompt = input_data.get(
        "prompt",
        "cinematic shot, ultra detailed, realistic, smooth camera movement"
    )
    negative = input_data.get(
        "negative_prompt",
        "low quality, blurry, deformed, text"
    )

    width = int(input_data.get("width", 512))
    height = int(input_data.get("height", 512))
    steps = int(input_data.get("steps", 18))
    guidance = float(input_data.get("guidance_scale", 7.5))

    num_frames = int(input_data.get("num_frames", 16))
    fps = int(input_data.get("fps", 8))

    seed = input_data.get("seed", None)

    pipe = _get_animatediff_pipeline()

    import torch
    import imageio.v2 as imageio

    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
    seed = int(seed)

    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    print(
        f"[ISE] Generando VIDEO AnimateDiff: prompt='{prompt[:60]}...' "
        f"({width}x{height}, steps={steps}, frames={num_frames}, fps={fps}, seed={seed})"
    )

    # Llamada típica a AnimateDiffPipeline
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )

    # result.frames suele ser List[List[PIL.Image]]
    frames = result.frames[0]

    # Guardar como MP4 en el volumen
    from datetime import datetime

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"isabela_ad_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = VIDEOS_DIR / name

    # Convertimos cada frame a RGB por si acaso
    frames_rgb = [f.convert("RGB") for f in frames]

    # imageio usa ffmpeg para MP4
    imageio.mimsave(
        out_path,
        frames_rgb,
        fps=fps,
    )

    print(f"[ISE] VIDEO AnimateDiff guardado en: {out_path}")

    video_b64 = _b64_from_file(out_path)

    return _ok(
        video_path=str(out_path),
        video_b64=video_b64,
        format="mp4",
        fps=fps,
        num_frames=len(frames_rgb),
        seed=seed,
        adapter_id=ANIMATEDIFF_ADAPTER_ID,
    )


=======
>>>>>>> 1e21153fc489de1f976887f457fa090274d29dc3
# ------------------ Handler de RunPod ------------------


def handler(event):
    """
    Formato típico:
    {
      "input": {
<<<<<<< HEAD
        "action": "image" | "video" | "animatediff" | "health",
=======
        "action": "image" | "video" | "health",
>>>>>>> 1e21153fc489de1f976887f457fa090274d29dc3
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

<<<<<<< HEAD
        # Video básico (GIF desde varios frames con SD normal)
        if action == "video":
            return generate_video_from_input(data)

        # Video con AnimateDiff (pipeline dedicado)
        if action == "animatediff":
            return generate_animatediff_video_from_input(data)

=======
        # Video básico (GIF desde varios frames)
        if action == "video":
            return generate_video_from_input(data)

>>>>>>> 1e21153fc489de1f976887f457fa090274d29dc3
        # Aquí después podemos ir agregando:
        # if action == "cinecam": ...
        # if action == "bodysync": ...
        # if action == "script2film": ...

        return _err(f"Unknown action '{action}'")

    except Exception as e:
        return _err(f"Exception: {e!r}")


runpod.serverless.start({"handler": handler})

