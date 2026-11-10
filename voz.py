import argparse
import sys
import wave
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import sounddevice as sd
from faster_whisper import WhisperModel

from config import ROOT


MODEL_PATH = ROOT / ".voice-models" / "base"
RECORD_SECONDS = 6


def record_sample(path: Path, device: int | None) -> None:
    info = sd.query_devices(device, "input")
    sample_rate = round(info["default_samplerate"])
    print(f"Micrófono: {info['name']}")
    input("Pulsa Enter cuando estés listo para hablar. ")

    samples = sd.rec(
        RECORD_SECONDS * sample_rate,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device,
    )
    print(f"Habla ahora. La grabación termina en {RECORD_SECONDS} segundos.", flush=True)
    sd.wait()

    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())


def transcribe(model: WhisperModel, path: Path) -> str:
    segments, _ = model.transcribe(str(path), language="es", vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


@cache
def load_model() -> WhisperModel:
    required_files = ("model.bin", "config.json", "tokenizer.json")
    if any(not (MODEL_PATH / name).is_file() for name in required_files):
        raise RuntimeError("Falta el modelo de voz. Consulta los pasos de instalación del README.")
    return WhisperModel(str(MODEL_PATH), device="cpu", compute_type="int8")


def recognize(audio: Path | None = None, device: int | None = None) -> tuple[str, float]:
    model = load_model()
    with TemporaryDirectory(prefix="nox-voz-") as directory:
        if audio is None:
            audio = Path(directory) / "microfono.wav"
            record_sample(audio, device)

        started = perf_counter()
        text = transcribe(model, audio)
        return text, perf_counter() - started


def review_voice_message() -> str | None:
    try:
        text, elapsed = recognize()
        if not text:
            print("No detecté voz. Escribe /voz para volver a intentarlo.")
            return None

        print(f"Transcripción ({elapsed:.1f} s): {text}")
        correction = input("Enter para enviar, escribe el texto corregido o /cancelar: ").strip()
        if correction.casefold() == "/cancelar":
            print("Mensaje de voz cancelado.")
            return None
        return correction or text
    except KeyboardInterrupt:
        print("\nMensaje de voz cancelado.")
    except (OSError, RuntimeError, sd.PortAudioError, ValueError) as error:
        print(f"No se pudo usar el micrófono: {error}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de transcripción local de Nox.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--devices", action="store_true", help="Lista los dispositivos de audio.")
    mode.add_argument("--file", type=Path, help="Transcribe un archivo en lugar de grabar.")
    parser.add_argument("--device", type=int, help="Número del micrófono en la lista.")
    args = parser.parse_args()

    try:
        if args.devices:
            print(sd.query_devices())
            return 0
        if args.file and not args.file.is_file():
            parser.error(f"No encuentro el archivo {args.file}.")
        text, elapsed = recognize(args.file, args.device)

        if text:
            print(f"Transcripción ({elapsed:.1f} s): {text}")
        else:
            print("No detecté voz. Comprueba el micrófono y vuelve a intentarlo.")
    except (KeyboardInterrupt, EOFError):
        print("\nPrueba de voz cancelada.")
    except (OSError, RuntimeError, sd.PortAudioError, ValueError) as error:
        print(f"No se pudo completar la prueba de voz: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
