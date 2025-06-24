"""
The parsing package is responsible for analyzing and extracting information
from Python source code. It uses Abstract Syntax Trees (AST) to deconstruct
code into a structured format, making it easier for other parts of the
application to understand and manipulate.
"""
from . import analysis, ast_parser, models
