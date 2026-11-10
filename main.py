import argparse
import json
import re
import sqlite3
import sys
from concurrent.futures import Future
from contextlib import closing

from brain import BrainError, reply
from config import DEFAULT_MODEL, PERSONALITY_PATH
from habla import DEFAULT_SPEECH_SPEED, SpeechError, speak
from memory import EXCERPT_LENGTH, conversation_context, open_history, save_turn


RECALL_CUE = re.compile(
    r"\b(?:recuerdas|recordar|hablamos|te (?:dije|cont[eé]|mencion[eé])|"
    r"(?:c[oó]mo se llama(?:ba)?|cu[aá]l (?:es|era) el nombre de) (?:mi|nuestr[oa]))\b",
    re.IGNORECASE,
)
SHORT_ACCEPTANCE = re.compile(
    r"(?:(?:bueno|vale|ok)[,\s]+)?(?:s[ií]|vale|dale|claro|ok|de acuerdo|adelante)",
    re.IGNORECASE,
)
WAIT_FOR_CONFIRMATION = re.compile(
    r"\bespera (?:a que te (?:diga que s[ií]|confirme)|mi confirmaci[oó]n)\b",
    re.IGNORECASE,
)
CONFIRMATION_REPLY = "De acuerdo. ¿Quieres que empiece?"
STORY_OFFER = re.compile(
    r"¿(?:te gustar[ií]a que te cuente|quieres que te cuente|te cuento)\b",
    re.IGNORECASE,
)
MISSED_STORY = re.compile(r"\bno me (?:la )?contaste\b.*\bla anterior\b", re.IGNORECASE)
CAPTURE_DESCRIPTION = re.compile(
    r"^\W*(?:qu[eé] (?:viste|ves)\W*$|"
    r"(?:qu[eé] (?:viste|ves|se ve|aparece)|descr[ií]be(?:me)?)\b.*"
    r"\b(?:captura|imagen|pantalla)\b)",
    re.IGNORECASE,
)
MAP_QUESTION = re.compile(r"\b(?:mapa|minimapa)\b", re.IGNORECASE)


def previous_story_offer(recent: list[dict[str, str]]) -> str:
    offers = list(dict.fromkeys(
        paragraph.strip()
        for message in reversed(recent)
        if message["role"] == "assistant"
        for paragraph in reversed(message["content"].split("\n\n"))
        if STORY_OFFER.search(paragraph)
    ))
    return offers[1] if len(offers) > 1 else next(iter(offers), "")


def chat_messages(
    history: sqlite3.Connection,
    personality: str,
    user_message: str,
    *,
    observation: str | None = None,
    looking: bool = False,
) -> list[dict[str, str]]:
    recall = RECALL_CUE.search(user_message)
    recent, older = conversation_context(history, user_message if recall else "")
    current = user_message
    if recall:
        excerpts = older + [
            message["content"][:EXCERPT_LENGTH]
            for message in recent
            if message["role"] == "user"
        ]
        recent = []
        personality += (
            "\nResponde a la pregunta de memoria en una frase usando solo los extractos. "
            "Si no está claro a cuál de varios datos se refiere el usuario, enumera las "
            "opciones que sí aparecen. Si el dato no aparece, di que no puedes confirmarlo."
        )
        if excerpts:
            history_json = json.dumps(excerpts, ensure_ascii=False)
            current = f"Extractos del usuario (JSON): {history_json}\n\nMensaje actual: {user_message}"
    elif recent and SHORT_ACCEPTANCE.fullmatch(user_message.strip(" ¡!.,…")):
        if recent[-1]["content"] == CONFIRMATION_REPLY:
            recent = recent[-2:]
        personality += (
            "\nEl mensaje actual es una respuesta breve a tu último turno. "
            "Si acepta una oferta o confirma una petición que quedó pendiente, "
            "entrégala directamente y termina ahí."
        )
    elif MISSED_STORY.search(user_message) and (offer := previous_story_offer(recent)):
        recent = [{"role": "assistant", "content": offer}]
        personality += (
            "\nLa oferta anterior que te piden retomar aparece en tu turno previo. "
            "Cuenta esa historia ahora, con un desenlace. Si es inventada, di que es ficción."
        )
    if not recall and observation:
        personality += (
            "\nLa observación visual es una descripción de una captura anterior y puede contener "
            "errores. Úsala solo si es pertinente al mensaje actual. Si te preguntan qué viste, "
            "responde en una o dos frases con los detalles visibles y el texto de la interfaz. "
            "Si no se puede confirmar que un personaje aparece, dilo. Distingue lo que el usuario "
            "contó de lo que muestra la captura. Termina ahí, sin suponer dónde está el personaje "
            "ni interpretar la vida o los sentimientos del usuario."
        )
        current = (
            f"Última observación visual (JSON): {json.dumps(observation, ensure_ascii=False)}"
            f"\n\nMensaje actual: {user_message}"
        )
        if CAPTURE_DESCRIPTION.search(user_message):
            recent = []
            personality = (
                "Eres Nox. Resume la observación visual en dos frases en español. "
                "Describe la escena o interfaz visible y conserva el texto relevante, "
                "como un marcador, un objetivo, un diálogo o una opción del menú. "
                "Solo usa nombres de lugares si aparecen como rótulos legibles. "
                "Presenta el marcador como resultado final solo si la pantalla indica que la partida terminó. "
                "Si preguntan por un personaje cuya presencia "
                "no está confirmada, di que la captura no permite confirmarla. "
                "Termina tras la descripción."
            )
            if not MAP_QUESTION.search(user_message):
                personality += " Deja fuera del resumen el mapa y cualquier interpretación del minimapa."
    elif not recall and looking:
        personality += "\nLa captura sigue en análisis. Si te preguntan qué viste, avisa que sigue pendiente."
    return [
        {"role": "system", "content": personality},
        *recent,
        {"role": "user", "content": current},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversa con Nox en tu PC.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo local de Ollama.")
    parser.add_argument("--hablar", action="store_true", help="Lee las respuestas en voz alta.")
    parser.add_argument("--cpu", action="store_true", help="Ejecuta también el modelo de conversación en CPU.")
    parser.add_argument(
        "--velocidad-voz", type=float, default=DEFAULT_SPEECH_SPEED,
        help="Ritmo de la voz, de 0.8 a 1.5 (predeterminado: %(default)s).",
    )
    args = parser.parse_args()
    if not 0.8 <= args.velocidad_voz <= 1.5:
        parser.error("--velocidad-voz debe estar entre 0.8 y 1.5.")

    try:
        personality = PERSONALITY_PATH.read_text(encoding="utf-8").strip()
        with closing(open_history()) as history:
            recent, _ = conversation_context(history)
            recovered = len(recent) // 2
            print(f"Nox · {args.model} · /voz para dictar · /mirar para capturar · /salir para cerrar.")
            if recovered:
                print(f"Intercambios recientes recuperados: {recovered}.")

            look: Future[str | None] | None = None
            while True:
                user_message = input("\nTú: ").strip()
                if user_message.lower() == "/salir":
                    break
                if user_message.lower() == "/mirar":
                    if look is not None and not look.done():
                        print("La captura anterior todavía se está analizando.")
                        continue
                    try:
                        from vision import start_look
                    except ImportError:
                        print(
                            r"Para usar la visión, abre Nox con .\.venv-voice\Scripts\python.exe main.py",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        look = start_look()
                    except (OSError, RuntimeError) as error:
                        print(f"No se pudo capturar el juego: {error}", file=sys.stderr)
                    continue
                if user_message.lower() == "/voz":
                    try:
                        from voz import review_voice_message
                    except ImportError:
                        print(
                            r"Para usar el micrófono, abre Nox con .\.venv-voice\Scripts\python.exe main.py",
                            file=sys.stderr,
                        )
                        continue
                    user_message = review_voice_message()
                if not user_message:
                    continue

                if WAIT_FOR_CONFIRMATION.search(user_message):
                    answer = CONFIRMATION_REPLY
                else:
                    looking = look is not None and not look.done()
                    observation = look.result() if look is not None and not looking else None
                    messages = chat_messages(
                        history, personality, user_message, observation=observation, looking=looking
                    )
                    print("Nox está preparando una respuesta…", flush=True)
                    try:
                        answer = reply(args.model, messages, cpu_only=args.cpu)
                    except BrainError as error:
                        print(f"Error: {error}", file=sys.stderr)
                        continue

                save_turn(history, user_message, answer)
                print(f"\nNox: {answer}", flush=True)
                if args.hablar:
                    try:
                        speak(answer, args.velocidad_voz)
                    except SpeechError as error:
                        print(f"Error de voz: {error}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        pass
    except (OSError, sqlite3.Error, UnicodeError) as error:
        print(f"Nox encontró un problema con sus archivos locales: {error}", file=sys.stderr)
        return 1

    print("\nNox cerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
