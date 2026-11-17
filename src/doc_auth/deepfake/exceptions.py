class DocAuthError(Exception):
    """Baza sənəd autentifikasiya xətası."""
    pass

class VideoLoadError(DocAuthError):
    """Video yüklənərkən və ya oxunarkən baş verən xəta."""
    pass

class SamplingLimitError(DocAuthError):
    """Frame limitləri aşdıqda baş verən xəta."""
    pass

class ModelUnavailableError(DocAuthError):
    """Model çeki (checkpoint) və ya ağırlıqları tapılmadıqda baş verən xəta."""
    pass