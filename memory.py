import re
import sqlite3
from collections.abc import Iterator
from heapq import nlargest

from config import DATABASE_PATH, OLDER_USER_TURNS, RECENT_TURNS


EXCERPT_LENGTH = 300
MAX_SEARCH_RESULTS = 4
WORD = re.compile(r"[^\W_]+")
ACCENTS = str.maketrans("áéíóúü", "aeiouu")
COMMON_WORDS = frozenset(
    "al algo como con cual cuales cuando de del donde el en era es esa ese esta este fue "
    "hablamos hay la las lo los llamaba llama me mi mis mencione nombre nos nuestra nuestro "
    "para por que quiero recuerdo recuerdas recordar se sobre su sus te tenia un una y yo "
    "contado conte dicho dije dijiste dime mencionamos mencionaste".split()
)


def open_history() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL
            )
            """
        )
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def search_older_messages(
    connection: sqlite3.Connection, before_id: int, question: str
) -> list[str]:
    terms = set(WORD.findall(question.lower().translate(ACCENTS))) - COMMON_WORDS
    if not terms:
        return []

    rows = connection.execute(
        "SELECT id, user_message FROM turns WHERE id < ?",
        (before_id,),
    )
    best = nlargest(MAX_SEARCH_RESULTS, matching_excerpts(rows, terms))
    return [excerpt for _, _, excerpt in sorted(best, key=lambda result: result[1])]


def matching_excerpts(
    rows: sqlite3.Cursor, terms: set[str]
) -> Iterator[tuple[int, int, str]]:
    for turn_id, message in rows:
        matched_terms = set()
        first_match = None
        for match in WORD.finditer(message):
            word = match.group().lower().translate(ACCENTS)
            if word in terms:
                matched_terms.add(word)
                if first_match is None:
                    first_match = match.start()
        if first_match is not None:
            start = max(0, first_match - EXCERPT_LENGTH // 3)
            yield len(matched_terms), turn_id, message[start : start + EXCERPT_LENGTH]


def conversation_context(
    connection: sqlite3.Connection, question: str = ""
) -> tuple[list[dict[str, str]], list[str]]:
    turn_limit = RECENT_TURNS + OLDER_USER_TURNS if question else RECENT_TURNS
    turns = connection.execute(
        """
        SELECT id, user_message, assistant_message FROM turns ORDER BY id DESC LIMIT ?
        """,
        (turn_limit,),
    ).fetchall()

    messages = []
    for _, user_message, assistant_message in reversed(turns[:RECENT_TURNS]):
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    older_excerpts = [
        user_message[:EXCERPT_LENGTH]
        for _, user_message, _ in reversed(turns[RECENT_TURNS:])
    ]
    if question and turns:
        older_excerpts = search_older_messages(connection, turns[-1][0], question) + older_excerpts
    return messages, older_excerpts


def save_turn(connection: sqlite3.Connection, user_message: str, answer: str) -> None:
    with connection:
        connection.execute(
            "INSERT INTO turns (user_message, assistant_message) VALUES (?, ?)",
            (user_message, answer),
        )
