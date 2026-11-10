# Nox

Compañero conversacional en español para Windows. Esta primera etapa
funciona desde PowerShell, con un modelo local de Ollama y un historial
en SQLite. El chat de texto usa la biblioteca estándar de Python.

## Arrancar Nox

Si tienes la versión portátil de Ollama en `.ollama`, abre PowerShell en
la carpeta del proyecto y ejecuta lo siguiente. Deja esta terminal abierta
mientras usas Nox.

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path (Get-Location) '.ollama\models'
$env:OLLAMA_NO_CLOUD = '1'
.\.ollama\ollama.exe serve
```

En otra terminal, desde la misma carpeta, descarga el modelo la primera
vez y abre Nox.

```powershell
.\.ollama\ollama.exe pull qwen3:4b-instruct-2507-q4_K_M
python main.py
```

Si prefieres la [instalación normal de Ollama para Windows](https://ollama.com/download/windows),
abre Ollama desde el menú Inicio y usa
`ollama pull qwen3:4b-instruct-2507-q4_K_M` en PowerShell.
En esa modalidad, puedes configurar
[`OLLAMA_NO_CLOUD=1`](https://docs.ollama.com/faq) antes de iniciar Ollama
para limitarlo a modelos locales.

Escribe `/salir` o pulsa `Ctrl+C` para cerrar Nox. Si arrancaste el servidor
portátil en su propia terminal, pulsa `Ctrl+C` allí para cerrarlo también.

## Probar la continuidad

1. Dile a Nox algo que quieras recordar, por ejemplo «Nuestra base se llama
   Faro Violeta».
2. Conversa un turno más y escribe `/salir`.
3. Vuelve a ejecutar `python main.py` y pregunta cómo se llama la base.

SQLite guarda todos los intercambios completados en `data/nox.db`. Para
una charla normal, Nox usa hasta ocho intercambios recientes. Ante una
pregunta directa de memoria, como «¿Cómo se llama mi nave?», consulta
hasta 16 mensajes tuyos y busca palabras como `nave` en el resto del
historial para recuperar hasta cuatro coincidencias antiguas. Una
pregunta vaga puede quedarse sin coincidencias. La ventana de contexto
es de 4.096 tokens y el historial completo queda guardado en SQLite.

## Probar el micrófono

La prueba de voz usa Whisper multilingüe en la CPU y muestra la transcripción
en español en la terminal. Funciona por separado del chat y de Ollama.

Si ya están instalados el entorno de voz y el modelo, ejecuta:

```powershell
.\.venv-voice\Scripts\python.exe voz.py
```

Pulsa **Enter** y, cuando salga «Habla ahora», di una frase corta.
La captura dura seis segundos y el archivo temporal se borra al terminar.

Si aparece un micrófono distinto al que quieres usar, lista los dispositivos
con `voz.py --devices` y vuelve a ejecutar el script con `--device N`,
sustituyendo `N` por el número del dispositivo de entrada.

Para preparar el entorno de voz en una instalación nueva, usa Python 3.13
y descarga el modelo una vez:

```powershell
py -3.13 -m venv .venv-voice
.\.venv-voice\Scripts\python.exe -m pip install -r requirements-voice.txt
.\.venv-voice\Scripts\python.exe -c "from faster_whisper.utils import download_model; download_model('base', output_dir='.voice-models/base')"
```

## Dictar un mensaje a Nox

Con Ollama abierto, inicia el chat desde el entorno de voz:

```powershell
.\.venv-voice\Scripts\python.exe main.py
```

Escribe `/voz`, pulsa Enter cuando lo pida y habla al ver «Habla ahora».
Revisa la transcripción antes de enviarla. Pulsa Enter para aceptarla,
escribe el mensaje corregido y pulsa Enter para enviarlo, o escribe
`/cancelar` para descartarlo. Solo el texto que envíes se guarda con la
respuesta en el historial. Nox responde en la terminal.

## Escuchar las respuestas de Nox

Con Ollama abierto, inicia Nox con la opción `--hablar`:

```powershell
.\.venv-voice\Scripts\python.exe main.py --hablar
```

Puedes dictar con `/voz` o escribir. Nox muestra su respuesta y luego la
lee por el dispositivo de salida predeterminado. Kokoro genera el audio
localmente con la voz Alex en español y un ritmo de `1.25`. Puedes
ajustarlo con `--velocidad-voz 1.35`. Para volver al chat escrito,
reinicia Nox sin `--hablar`.

En una instalación nueva, tras instalar `requirements-voice.txt`,
descarga el modelo y el archivo de voces:

```powershell
New-Item -ItemType Directory -Force .voice-models/kokoro
$kokoro = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0'
curl.exe -fL "$kokoro/kokoro-v1.0.onnx" -o .voice-models/kokoro/kokoro-v1.0.onnx
curl.exe -fL "$kokoro/voices-v1.0.bin" -o .voice-models/kokoro/voices-v1.0.bin
```

## Probar la visión con el juego

Con Ollama abierto, ejecuta:

```powershell
.\.venv-voice\Scripts\python.exe vision.py
```

Pulsa Enter, cambia a la ventana del juego dentro de cinco segundos y
déjala en primer plano unos segundos después de que termine la cuenta
regresiva. Vuelve a PowerShell para leer la descripción y el tiempo de
análisis. Si la captura directa sale negra, Nox usa los píxeles visibles
de la región que ocupa el juego en pantalla. Si ambos métodos fallan,
puedes analizar una captura guardada con
`vision.py --file "C:\ruta\captura.png"` usando el mismo Python del
entorno de voz.

La prueba usa Qwen3-VL 2B en Ollama local. Toma una imagen de la ventana
activa, la mantiene en memoria y pide liberar el modelo tras responder.
El tiempo que muestra incluye la carga del modelo y el análisis, después
de la cuenta regresiva.

Para comparar dos capturas distintas seguidas, añade `--mantener-modelo`
a ambos comandos y haz la segunda en menos de dos minutos. El modelo
queda cargado entre solicitudes. La línea «Tiempos de Ollama» separa
la carga, el procesamiento de la entrada y la redacción. Puedes consultar
cuánto del modelo se ejecuta en GPU con `.\.ollama\ollama.exe ps`
mientras sigue cargado.

Para hacer la inferencia visual en CPU durante el juego o una transmisión,
ejecuta `.\.venv-voice\Scripts\python.exe vision.py --cpu`. La captura y
los tiempos funcionan igual. La opción mueve el análisis de Ollama a la CPU
y puede competir con el juego por tiempo de procesador.

También puedes usar la visión desde el chat. Abre Nox con el Python del
entorno de voz y escribe `/mirar`. Pulsa Enter, cambia al juego y deja la
ventana en primer plano mientras se toma la captura. El análisis en CPU
sigue en segundo plano y la descripción aparece en la terminal. Puedes
escribir mientras tanto, aunque Ollama puede tardar en atender ambos
modelos a la vez. Cuando aparezca la descripción, pregúntale a Nox qué vio.
La captura se mantiene en memoria y la descripción queda disponible
durante esa sesión.

La misma prueba sirve para distintos juegos. Nox describe la escena y el
texto legible, ya sea un objetivo, un marcador, un diálogo o un menú.
Si quieres hablar de las reglas de un juego, puedes decirle cuál es.

Si quieres que el modelo de conversación también use la CPU mientras
juegas, inicia Nox con `.\.venv-voice\Scripts\python.exe main.py --cpu`.
Puedes combinarlo con `--hablar` para escuchar las respuestas.

En una instalación nueva, prepara el entorno y descarga el modelo una vez:

```powershell
.\.venv-voice\Scripts\python.exe -m pip install Pillow==13.0.0
.\.ollama\ollama.exe pull qwen3-vl:2b-instruct
```

## Ajustar personalidad y modelo

- Edita `personality.txt` y reinicia Nox para probar cambios de tono.
- Para probar otro modelo local ya descargado, usa
  `python main.py --model NOMBRE_DEL_MODELO`.
- `config.py` define el modelo predeterminado y el tamaño del contexto.
  `brain.py` se comunica con Ollama, `memory.py` guarda y recupera los
  intercambios y `main.py` coordina el chat.
