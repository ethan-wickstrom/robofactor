import contextlib

with contextlib.suppress(Exception):
    import os

    from beartype import BeartypeConf
    from beartype.claw import beartype_all, beartype_this_package
    if os.environ.get("ROBOFACTOR_BEARTYPE_THIS_PACKAGE", "0") == "1":
        beartype_this_package()
    if os.environ.get("ROBOFACTOR_BEARTYPE_ALL", "0") == "1":
        beartype_all(conf=BeartypeConf(violation_type=UserWarning))
from .data import examples, models

__all__: list[str] = ["examples", "models"]
