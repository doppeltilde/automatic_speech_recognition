# Automatic Speech Recognition utilizing Faster Whisper.

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [Python](https://www.python.org)
- [Docker](https://docker.com)

## Installation

- For ease of use it's recommended to use the provided [docker-compose.yml](https://github.com/doppeltilde/automatic_speech_recognition/blob/main/docker-compose.yml).
**CPU Support:** Use the `latest` tag.
```yml
services:
  automatic_speech_recognition:
    image: ghcr.io/doppeltilde/automatic_speech_recognition:latest
    ports:
      - "8000:8000"
    volumes:
      - models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_ASR_MODEL_NAME
      - COMPUTE_TYPE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped

volumes:
  models:
```

**NVIDIA GPU Support:** Use the `latest-cuda` tag.
```yml
services:
  automatic_speech_recognition_cuda:
    image: ghcr.io/doppeltilde/automatic_speech_recognition:latest-cuda
    ports:
      - "8000:8000"
    volumes:
      - models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_ASR_MODEL_NAME
      - COMPUTE_TYPE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]

volumes:
  models:
```


- Create a `.env` file and set the preferred values.
```sh
DEFAULT_ASR_MODEL_NAME=base
COMPUTE_TYPE=float16

# False == Public Access
# True == Access Only with API Key
USE_API_KEYS=False

# Comma seperated api keys
API_KEYS=abc,123,xyz
```

## Models
Any model designed and compatible with faster-whisper should work.

## Usage

> [!NOTE]
> Please be aware that the initial process may require some time, as the model is being downloaded.

> [!TIP]
> Interactive API documentation can be found at: http://localhost:8000/docs

---

_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._