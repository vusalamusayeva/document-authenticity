class DeepfakeError(Exception):
    """Deepfake modulu üçün əsas xəta sinfi."""


class InvalidConfigError(DeepfakeError, ValueError):
    """Etibarsız konfiqurasiya və ya skor/çəki dəyəri xətası."""


class ModelUnavailableError(DeepfakeError):
    """Model çəkisi və ya detektor əlçatan olmadıqda qaldırılan xəta."""


class VideoValidationError(DeepfakeError):
    """Video faylı və ya kadr axını etibarsız olduqda qaldırılan xəta."""


class VideoLoadError(VideoValidationError):
    """Video faylının açılması və ya oxunması zamanı baş verən xəta."""


class FrameSamplingError(VideoValidationError):
    """Kadrların seçilməsi (sampling) zamanı baş verən xəta."""