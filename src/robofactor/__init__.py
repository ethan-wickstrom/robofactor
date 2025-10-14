import contextlib

with contextlib.suppress(Exception):
    import os

    if os.environ.get("ROBOFACTOR_ENABLE_BEARTYPE") == "1":
        from beartype import BeartypeConf
        from beartype.claw import beartype_all, beartype_this_package

        beartype_this_package()

        if os.environ.get("ROBOFACTOR_BEARTYPE_ALL") == "1":
            beartype_all(conf=BeartypeConf(violation_type=UserWarning))

from .data import examples, models

__all__ = ["examples", "models"]
