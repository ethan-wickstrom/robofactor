from typing import Annotated
from pydantic import Field

PositiveInt = Annotated[int, Field(gt=0)]
Temperature = Annotated[float, Field(ge=0.0, le=2.0)]
