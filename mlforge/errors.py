class ForgeError(Exception):
    """A safe, actionable message that may be returned to a local user."""

    def __init__(self, message: str, code: str = "invalid_input", status: int = 422):
        super().__init__(message)
        self.code, self.status = code, status
