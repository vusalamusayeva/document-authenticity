"""
İnference pipeline-ı çağıran servis qatı.
API route-ları birbaşa model kodunu çağırmamalı — hər zaman bu qatdan keçməlidir.
"""

import uuid

from api.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse


class AnalysisService:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline  # doc_auth.inference.pipeline.Pipeline instansı

    def analyze(self, request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
        raise NotImplementedError(
            "1. base64 -> image decode et\n"
            "2. self.pipeline.run(image) çağır\n"
            "3. nəticəni DocumentAnalysisResponse-a map et"
        )
        # nümunə skelet:
        # request_id = str(uuid.uuid4())
        # result = self.pipeline.run(image)
        # return DocumentAnalysisResponse(request_id=request_id, **result)
