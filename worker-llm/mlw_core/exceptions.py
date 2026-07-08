class BusinessLogicException(Exception):
    """Base class for business logic exceptions"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class MLTaskProcessingFailed(BusinessLogicException):
    """Insufficient credits on the credit balance"""
    def __init__(self):
        super().__init__(
            message = 'ML model failed to process task'
        )