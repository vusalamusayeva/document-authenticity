class SignatureStampError(Exception):
    """İmza və mühür modulu üçün təməl xəta sinfi."""
    pass

class SignatureNotFoundError(SignatureStampError):
    """Sənəddə imza tapılmadıqda yaranan xəta."""
    pass

class StampAnalysisError(SignatureStampError):
    """Mühür analizi zamanı xəta baş verdikdə."""
    pass