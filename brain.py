import json
from base64 import b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException

from config import CONTEXT_TOKENS, RESPONSE_TOKENS


class BrainError(Exception):
    """Error de comunicación con el motor local."""


@dataclass(frozen=True)
class ChatResponse:
    text: str
    load_ns: int | None
    input_ns: int | None
    generation_ns: int | None


def reply(
    model: str, messages: list[dict[str, str]], *,
    image: bytes | None = None, cpu_only: bool = False,
) -> str:
    return request_chat(
        model, messages, image=image, keep_alive=0 if image is not None else "1m", cpu_only=cpu_only
    ).text


def request_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    image: bytes | None = None,
    keep_alive: str | int = "1m",
    cpu_only: bool = False,
) -> ChatResponse:
    chat_messages: Sequence[Mapping[str, str | list[str]]] = messages
    options: dict[str, int | float] = {"num_ctx": CONTEXT_TOKENS, "num_predict": RESPONSE_TOKENS}
    if image is not None:
        encoded = b64encode(image).decode("ascii")
        chat_messages = [*messages[:-1], {**messages[-1], "images": [encoded]}]
        options["temperature"] = 0.2
    if cpu_only:
        options["num_gpu"] = 0

    payload = json.dumps(
        {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": options,
        }
    ).encode("utf-8")

    connection = HTTPConnection("127.0.0.1", 11434, timeout=120)
    try:
        connection.request(
            "POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        body = response.read()
    except ConnectionRefusedError as error:
        raise BrainError("Ollama no está activo. Inícialo y vuelve a enviar el mensaje.") from error
    except TimeoutError as error:
        raise BrainError("Ollama tardó demasiado en responder. Inténtalo de nuevo.") from error
    except (OSError, HTTPException) as error:
        raise BrainError(f"Falló la conexión con Ollama: {error}") from error
    finally:
        connection.close()

    if response.status == 404:
        raise BrainError(f"Ollama no encontró el modelo {model!r}. Descárgalo con ollama pull.")

    if response.status != 200:
        raise BrainError(f"Ollama devolvió un error HTTP {response.status}. Revisa su consola.")

    try:
        data = json.loads(body)
        answer = data["message"]["content"].strip()
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise BrainError("Ollama devolvió una respuesta que Nox no pudo leer.") from error

    if not answer:
        raise BrainError("El modelo devolvió una respuesta vacía. Inténtalo de nuevo.")

    return ChatResponse(
        text=answer,
        load_ns=data.get("load_duration"),
        input_ns=data.get("prompt_eval_duration"),
        generation_ns=data.get("eval_duration"),
    )
