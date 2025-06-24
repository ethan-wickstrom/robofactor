import warnings


def suppress_pydantic_warnings():
    # TODO: Remove this warning suppression once DSPy fixes Pydantic serialization compatibility.
    #
    # This is really frustrating me and this is a terrible solution. However, DSPy seems to working and
    # since it's just a warning, I'll embrace the blissful ignorance.
    #
    # Nevertheless, this is a temporary solution until DSPy fixes Pydantic serialization compatibility.
    #
    # I might just be dumb, but I couldn't find a clean solution to fix these Pydantic serialization warnings
    # that appear every time DSPy interacts with OpenAI/LiteLLM responses. The warnings look like this:
    #
    #   ```bash
    #   UserWarning: Pydantic serializer warnings:
    #   PydanticSerializationUnexpectedValue(Expected 9 fields but got 5: Expected `Message` - serialized value may not be as expected [input_value=Message(content='{\n  "an...er_specific_fields=None), input_type=Message])
    #   PydanticSerializationUnexpectedValue(Expected `StreamingChoices` - serialized value may not be as expected [input_value=Choices(finish_reason='st...r_specific_fields=None)), input_type=Choices])
    #   return self.__pydantic_serializer__.to_python(
    #   ```
    #
    # WHEN TO REMOVE THIS:
    # 1. Check DSPy GitHub issues/releases for fixes to Pydantic serialization
    #    (search for "PydanticSerializationUnexpectedValue" or something like this)
    # 2. Try removing this suppression after DSPy updates (especially major/minor version bumps)
    # 3. If warnings persist, check if we're using the correct LM configuration (e.g., dspy.LM vs legacy clients)
    #
    # If the three steps above don't work, keep the warning suppression.
    #
    # TRACKING:
    # - DSPy Issue: https://github.com/stanfordnlp/dspy/issues (search for Pydantic serialization)
    #
    # Last checked: 19 June 2025
    # DSPy version when added: 2.6.27
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
