from functools import wraps


def require_fit(func):
    """Decorator to ensure the model has been fitted before calling a method.

    Raises ValueError if the model hasn't been fitted yet.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fitted_:
            raise ValueError(
                "This model instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        return func(self, *args, **kwargs)

    return wrapper
