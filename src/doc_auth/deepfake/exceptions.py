class DocAuthError(Exception):
    """Baza xəta sinfi."""
    pass

class VideoLoadError(DocAuthError):
    """Video yüklənərkən və ya oxunarkən baş verən xətalar."""
    pass

class ModelUnavailableError(DocAuthError):
    """Model mövcud olmadıqda və ya yüklənmədikdə."""
    pass

class InvalidConfigError(DocAuthError):
    """Yanlış və ya etibarsız konfiqurasiya daxil edildikdə."""
    pass