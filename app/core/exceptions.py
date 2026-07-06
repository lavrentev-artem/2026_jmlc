from fastapi import status

class BusinessLogicException(Exception):
    """Base class for business logic exceptions"""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class InsufficientCredits(BusinessLogicException):
    """Insufficient credits on the credit balance"""
    def __init__(self):
        super().__init__(
            message = 'Insufficient credits',
            status_code = status.HTTP_402_PAYMENT_REQUIRED
        )


class OperationFailed(BusinessLogicException):
    """Failed to perform the operation"""

    def __init__(self):
        super().__init__(
            message='Operation failed',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

class MLPromptSendingFailed(BusinessLogicException):
    """Failed to send prompt to ML service"""

    def __init__(self):
        super().__init__(
            message='ML prompt failed',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class MLPromptFailed(BusinessLogicException):
    """Failed to product output on the ML prompt"""

    def __init__(self):
        super().__init__(
            message='ML prompt failed',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

