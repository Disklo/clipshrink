# bad flags, missing tool, unreadable image
class ClipshrinkError(Exception):
    pass


# backend or encoder failed, RuntimeError for old handlers
class BackendError(ClipshrinkError, RuntimeError):
    pass
