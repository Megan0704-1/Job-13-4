from gymnasium import Wrapper


class ForwardMethodWrapper(Wrapper):
    _reserved = {"env", "metadata", "action_space"}
    _forwarded = set()

    def __getattr__(self, name):
        if name in self._reserved:
            raise AttributeError(f"Accessing reserved attribute `{name}`")

        env = self.env
        seen = set()

        while env not in seen:
            seen.add(env)
            if hasattr(env, name):
                self._forwarded.add(name)
                return getattr(env, name)
            if hasattr(env, "env"):
                env = env.env
            else:
                break

        raise AttributeError(
            f"'{type(self).__name__}' and nested environments "
            f"have no attribute '{name}'"
        )

    def __dir__(self):
        """Include forwarded methods for autocompletion"""
        return super().__dir__() + list(self._forwarded)
