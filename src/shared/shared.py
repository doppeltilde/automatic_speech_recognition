from dotenv import load_dotenv
import os
from faster_whisper import WhisperModel
import torch

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)

default_asr_model_name = os.getenv("DEFAULT_ASR_MODEL_NAME", "base")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = os.getenv("COMPUTE_TYPE", "int8")

# API KEY
api_keys_str = os.getenv("API_KEYS", "")
api_keys = api_keys_str.split(",") if api_keys_str else []
use_api_keys = os.getenv("USE_API_KEYS", "False").lower() in ["true", "1", "yes"]


def asr_model(model_name):
    try:
        _model_name = model_name or default_asr_model_name

        model = WhisperModel(
            _model_name,
            device=device,
            compute_type=compute_type,
        )

        return model

    except Exception as e:
        print(e)
        return {"error": str(e)}
