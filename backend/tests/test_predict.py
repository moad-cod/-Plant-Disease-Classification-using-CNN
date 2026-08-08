import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.api.predict import predict


def test_predict_rejects_non_image_upload():
    upload = UploadFile(filename="notes.txt", file=BytesIO(b"not an image"))
    upload.headers = {"content-type": "text/plain"}

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(predict(upload))

    assert exc_info.value.status_code == 400
    assert "Only JPG" in exc_info.value.detail
