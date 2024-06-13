from fastapi import APIRouter, UploadFile, File, Query, Depends
from src.middleware.auth.auth import get_api_key
from src.shared.shared import asr_model
import io
import traceback
import time
from src.shared.shared import LANGUAGE_CODES
import gc
import torch

router = APIRouter()


@router.post("/api/asr", dependencies=[Depends(get_api_key)])
async def asr(
    file: UploadFile = File(),
    model_name: str = Query(None),
    word_timestamps: bool = Query(False),
    beam_size: int = Query(1),
    language: str = Query(
        None,
        enum=LANGUAGE_CODES,
        description="If not provided, will fallback to detected language.",
    ),
):

    start_time = time.time()
    model = asr_model(model_name)

    try:
        content = await file.read()
        audio_file = io.BytesIO(content)

        segments, info = model.transcribe(
            audio_file,
            beam_size=beam_size,
            word_timestamps=True if word_timestamps else False,
            language=language,
        )

        segment_words = []
        entire_transcription = ""

        for index, segment in enumerate(segments):
            if index == 0:
                trimmed_segment_text = segment.text.strip()
            else:
                trimmed_segment_text = segment.text
            entire_transcription += trimmed_segment_text

            if segment.words is not None:
                for word in segment.words:
                    segment_words.append(
                        {"start": word.start, "end": word.end, "word": word.word}
                    )

        return {
            "res": {
                "execution_time": time.time() - start_time,
                "language": info.language,
                "language_probability": info.language_probability,
                "text": entire_transcription,
                "words": segment_words,
            }
        }

    except Exception as e:
        print("Something went wrong: ", e)
        return {"error": str(traceback.format_exc())}

    finally:
        audio_file.close()
        del model
        torch.cuda.empty_cache()
        gc.collect()
