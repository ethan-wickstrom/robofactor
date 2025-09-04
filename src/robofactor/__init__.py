try:
    import os

    from beartype import BeartypeConf
    from beartype.claw import beartype_all, beartype_this_package

    beartype_this_package()

    if os.environ.get("ROBOFACTOR_BEARTYPE_ALL", "1") == "1":
        beartype_all(conf=BeartypeConf(violation_type=UserWarning))
except Exception:
    pass

from .data import examples, models

__all__ = ["examples", "models"]
