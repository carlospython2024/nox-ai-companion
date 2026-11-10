from functools import cache

from config import ROOT


MODEL_DIR = ROOT / ".voice-models" / "kokoro"
VOICE_NAME = "em_alex"
DEFAULT_SPEECH_SPEED = 1.25


class SpeechError(Exception):
    """Error al reproducir la respuesta de Nox."""


@cache
def load_synthesizer():
    from kokoro_onnx import Kokoro

    model = MODEL_DIR / "kokoro-v1.0.onnx"
    voices = MODEL_DIR / "voices-v1.0.bin"
    if not model.is_file() or not voices.is_file():
        raise SpeechError("Falta Kokoro. Consulta la instalación de la voz en el README.")
    return Kokoro(str(model), str(voices))


def speak(text: str, speed: float = DEFAULT_SPEECH_SPEED) -> None:
    try:
        import sounddevice as sd

        samples, sample_rate = load_synthesizer().create(
            text, voice=VOICE_NAME, lang="es-419", speed=speed
        )
        if samples.size:
            sd.play(samples, sample_rate, blocking=True)
    except ImportError as error:
        raise SpeechError(
            r"Usa el entorno de voz: .\.venv-voice\Scripts\python.exe main.py --hablar"
        ) from error
    except SpeechError:
        raise
    except Exception as error:
        raise SpeechError(f"No se pudo generar o reproducir la respuesta: {error}") from error
