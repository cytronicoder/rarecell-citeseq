import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    import numba

    _original_njit = numba.njit


    def _njit_without_cache(*args, **kwargs):
        kwargs["cache"] = False
        return _original_njit(*args, **kwargs)


    numba.njit = _njit_without_cache
except Exception:
    pass
