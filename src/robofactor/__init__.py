import contextlib
import os

if os.environ.get("ROBOFACTOR_ENABLE_BEARTYPE") == "1":
    with contextlib.suppress(Exception):
        from beartype.claw import beartype_this_package

        beartype_this_package()

        if os.environ.get("ROBOFACTOR_BEARTYPE_ALL") == "1":
            from beartype import BeartypeConf
            from beartype.claw import beartype_all

            beartype_all(conf=BeartypeConf(violation_type=UserWarning))

from .data import examples, models

__all__ = ["examples", "models"]
