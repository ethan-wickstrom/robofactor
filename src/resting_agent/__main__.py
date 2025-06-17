import dspy
from .core.config import AppConfig


def main():
    """Main entry point for demonstrating the Laravel API Agent."""
    # Configure DSPy with a capable model and ChatAdapter for structured output.
    lm = dspy.LM('openai/gpt-4o-mini', max_tokens=4096)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter)


if __name__ == '__main__':
    main()
