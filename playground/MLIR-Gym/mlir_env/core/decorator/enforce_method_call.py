# decorator/enforce_method_call


# enforce call to method name before proceeding
# must set _{method_name}_called variable in the required called method
def enforce_call(method_name):
    def decorator(method):
        def wrapper(instance, *args, **kwargs):
            if not getattr(instance, f"_{method_name}_called", False):
                raise RuntimeError(
                    f"You must call {method_name} before calling {method.__name__}"
                )

            return method(instance, *args, **kwargs)

        return wrapper

    return decorator
