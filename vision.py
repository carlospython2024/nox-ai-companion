import argparse
import ctypes
import sys
from concurrent.futures import Future
from ctypes import wintypes
from io import BytesIO
from pathlib import Path
from threading import Thread
from time import perf_counter, sleep

from PIL import Image, ImageGrab

from brain import BrainError, ChatResponse, request_chat


VISION_MODEL = "qwen3-vl:2b-instruct"
DWMWA_EXTENDED_FRAME_BOUNDS = 9
VISION_PROMPT = (
    "Describe en español la escena o interfaz visible en dos o tres frases. "
    "Si hay texto legible, cita uno o dos datos relevantes, como un marcador, un objetivo, "
    "un diálogo o una opción de menú. En mapas e iconos, distingue las formas visibles de los "
    "rótulos. Menciona un lugar solo si su nombre aparece escrito en la imagen. "
    "Un marcador solo es un resultado final si la pantalla indica que la partida terminó. "
    "Evita adivinar el juego, sus reglas o nombres que no se distinguen."
)


def window_bounds(window: int) -> tuple[int, int, int, int]:
    bounds = wintypes.RECT()
    get_attribute = ctypes.windll.dwmapi.DwmGetWindowAttribute
    get_attribute.argtypes = (
        wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD
    )
    get_attribute.restype = ctypes.c_long
    result = get_attribute(
        window, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(bounds), ctypes.sizeof(bounds)
    )
    if result != 0 or bounds.right <= bounds.left or bounds.bottom <= bounds.top:
        raise RuntimeError("Windows no pudo determinar la región visible de la ventana.")
    return bounds.left, bounds.top, bounds.right, bounds.bottom


def capture_window() -> Image.Image:
    foreground = ctypes.windll.user32.GetForegroundWindow
    foreground.restype = wintypes.HWND
    console_window = foreground()
    input("Pulsa Enter y cambia a la ventana del juego. ")
    for remaining in range(5, 0, -1):
        print(f"Captura en {remaining}…", flush=True)
        sleep(1)

    game_window = foreground()
    if not game_window or game_window == console_window:
        raise RuntimeError("No detecté el cambio de ventana. Vuelve a intentarlo y abre el juego.")
    try:
        image = ImageGrab.grab(window=game_window)
    except OSError:
        image = None
    if image is not None and image.convert("RGB").getbbox():
        return image

    # Some games render correctly on the desktop but return black in window captures.
    bounds = window_bounds(game_window)
    if foreground() != game_window:
        raise RuntimeError("El juego perdió el foco antes de la captura. Déjalo en primer plano unos segundos más.")
    image = ImageGrab.grab(bbox=bounds, all_screens=True)
    if foreground() != game_window:
        raise RuntimeError("El juego perdió el foco durante la captura. Vuelve a intentarlo.")
    if not image.convert("RGB").getbbox():
        raise RuntimeError("La captura sigue negra. Guarda una captura desde el juego y analízala con --file.")
    return image


def encode_image(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()


def analyze_image(
    image: bytes, *, cpu_only: bool, keep_alive: str | int = 0
) -> ChatResponse:
    return request_chat(
        VISION_MODEL,
        [{"role": "user", "content": VISION_PROMPT}],
        image=image,
        keep_alive=keep_alive,
        cpu_only=cpu_only,
    )


def start_look() -> Future[str | None]:
    image = encode_image(capture_window())
    result: Future[str | None] = Future()
    print("Captura tomada. Nox la está analizando en CPU. Puedes seguir escribiendo.", flush=True)

    def analyze() -> None:
        description = None
        started = perf_counter()
        try:
            description = analyze_image(image, cpu_only=True).text
            print(f"\nNox · captura ({perf_counter() - started:.1f} s): {description}", flush=True)
        except BrainError as error:
            print(f"\nNo se pudo analizar la captura: {error}", file=sys.stderr, flush=True)
        finally:
            result.set_result(description)

    Thread(target=analyze, name="nox-vision", daemon=True).start()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de visión local de Nox.")
    parser.add_argument("--file", type=Path, help="Analiza un archivo en lugar de capturar una ventana.")
    parser.add_argument("--cpu", action="store_true", help="Ejecuta el análisis visual en CPU.")
    parser.add_argument(
        "--mantener-modelo", action="store_true",
        help="Mantiene el modelo cargado dos minutos para repetir la prueba.",
    )
    args = parser.parse_args()

    try:
        if args.file:
            with Image.open(args.file) as source:
                image = encode_image(source)
        else:
            image = encode_image(capture_window())
        mode = " en CPU" if args.cpu else ""
        print(f"Nox está analizando la imagen{mode}…", flush=True)
        started = perf_counter()
        answer = analyze_image(
            image,
            keep_alive="2m" if args.mantener_modelo else 0,
            cpu_only=args.cpu,
        )
        elapsed = perf_counter() - started
        print(f"\nNox ({elapsed:.1f} s): {answer.text}")
        timings = [
            f"{label} {duration / 1_000_000_000:.1f} s"
            for label, duration in (
                ("carga", answer.load_ns),
                ("procesar entrada", answer.input_ns),
                ("redactar", answer.generation_ns),
            )
            if duration is not None
        ]
        if timings:
            print("Tiempos de Ollama · " + " · ".join(timings))
    except (KeyboardInterrupt, EOFError):
        print("\nPrueba de visión cancelada.")
    except (BrainError, OSError, RuntimeError, Image.DecompressionBombError) as error:
        print(f"No se pudo analizar la imagen: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
