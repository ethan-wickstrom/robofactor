from collections.abc import Mapping, Sequence

type JSONPrimitive = None | bool | int | float | str
type JSONSequence = Sequence["JSON"]
type JSONObject = Mapping[str, "JSON"]
type JSON = JSONPrimitive | JSONSequence | JSONObject
