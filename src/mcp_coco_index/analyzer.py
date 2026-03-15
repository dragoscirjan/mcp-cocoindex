"""Tree-sitter based code analyzer for structural symbol extraction.

This module provides symbol extraction capabilities using tree-sitter for parsing
and language-specific queries for identifying definitions and references.

Supported languages (18 programming languages with tree-sitter grammars):
- Python, JavaScript, TypeScript, TSX
- Go, Rust, Java, C, C++, C#
- Ruby, PHP, Swift, Kotlin, Scala
- R, Pascal, Fortran, Solidity

For languages without tree-sitter grammars, an LLM fallback can be used (opt-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a code symbol (definition or reference)."""

    name: str
    """The identifier name (e.g., 'MyClass', 'process_data', 'my_property')."""

    kind: str
    """Symbol kind: class, function, method, variable, attribute, import, parameter."""

    category: str
    """'definition' or 'reference'."""

    line: int
    """1-based line number."""

    column: int
    """0-based column number."""

    end_line: int
    """End line (1-based)."""

    end_column: int
    """End column (0-based)."""

    context: str
    """The source line containing the symbol (for display)."""

    parent_name: str | None = None
    """Enclosing class/function name, or object name for attribute access."""

    parent_kind: str | None = None
    """Kind of parent ('class', 'function', etc.)."""


# Extension to tree-sitter language name mapping
# Based on CocoIndex's internal prog_langs.rs mapping
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    ".pyx": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "tsx",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java
    ".java": "java",
    # C
    ".c": "c",
    # C++
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".h": "cpp",  # Could be C or C++, default to C++
    # C#
    ".cs": "c_sharp",
    # Ruby
    ".rb": "ruby",
    # PHP
    ".php": "php",
    # Swift
    ".swift": "swift",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
    # Scala
    ".scala": "scala",
    # R
    ".r": "r",
    ".R": "r",
    # Pascal
    ".pas": "pascal",
    ".dpr": "pascal",
    # Fortran
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    # Solidity
    ".sol": "solidity",
}

# Languages that have tree-sitter grammars via tree-sitter-language-pack
TREE_SITTER_LANGUAGES: set[str] = {
    "python",
    "javascript",
    "typescript",
    "tsx",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "c_sharp",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "r",
    "pascal",
    "fortran",
    "solidity",
}

# Data/markup formats - skip symbol extraction for these
SKIP_LANGUAGES: set[str] = {
    "json",
    "yaml",
    "toml",
    "xml",
    "html",
    "css",
    "markdown",
    "sql",
    "dtd",
}


def detect_language(file_path: str) -> str | None:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Tree-sitter language name, or None if unknown/unsupported
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def has_tree_sitter_support(language: str | None) -> bool:
    """Check if a language has tree-sitter grammar support."""
    return language is not None and language in TREE_SITTER_LANGUAGES


def should_skip_language(language: str | None) -> bool:
    """Check if a language should be skipped (data/markup formats)."""
    return language is not None and language in SKIP_LANGUAGES


# Tree-sitter query patterns for each language
# These extract definitions and references for classes, functions, methods, variables, etc.

PYTHON_QUERIES = """
; Class definitions
(class_definition
  name: (identifier) @class.definition)

; Function definitions (top-level)
(function_definition
  name: (identifier) @function.definition)

; Method definitions (inside class)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.definition)))

; Variable assignments (simple identifier on left)
(assignment
  left: (identifier) @variable.definition)

; Attribute assignments (self.x = ...)
(assignment
  left: (attribute
    object: (identifier) @attr.object
    attribute: (identifier) @attribute.definition))

; Function parameters
(parameters
  (identifier) @parameter.definition)
(parameters
  (typed_parameter
    (identifier) @parameter.definition))
(parameters
  (default_parameter
    name: (identifier) @parameter.definition))

; Import statements
(import_statement
  name: (dotted_name) @import.definition)
(import_from_statement
  module_name: (dotted_name) @import.definition)
(import_from_statement
  name: (dotted_name
    (identifier) @import.definition))

; Function calls
(call
  function: (identifier) @function.reference)
(call
  function: (attribute
    attribute: (identifier) @method.reference))

; Attribute access (reading)
(attribute
  attribute: (identifier) @attribute.reference)

; Identifier references (variable reads, type annotations)
(identifier) @identifier.reference
"""

JAVASCRIPT_QUERIES = """
; Class declarations
(class_declaration
  name: (identifier) @class.definition)

; Function declarations
(function_declaration
  name: (identifier) @function.definition)

; Arrow functions assigned to variables
(variable_declarator
  name: (identifier) @function.definition
  value: (arrow_function))

; Method definitions
(method_definition
  name: (property_identifier) @method.definition)

; Variable declarations
(variable_declarator
  name: (identifier) @variable.definition)

; Parameters
(formal_parameters
  (identifier) @parameter.definition)
(formal_parameters
  (assignment_pattern
    left: (identifier) @parameter.definition))

; Import statements
(import_specifier
  name: (identifier) @import.definition)
(import_clause
  (identifier) @import.definition)

; Function calls
(call_expression
  function: (identifier) @function.reference)
(call_expression
  function: (member_expression
    property: (property_identifier) @method.reference))

; Property access
(member_expression
  property: (property_identifier) @attribute.reference)

; Identifier references
(identifier) @identifier.reference
"""

TYPESCRIPT_QUERIES = (
    JAVASCRIPT_QUERIES
    + """
; Interface declarations
(interface_declaration
  name: (type_identifier) @class.definition)

; Type alias declarations
(type_alias_declaration
  name: (type_identifier) @class.definition)

; Property signatures in interfaces
(property_signature
  name: (property_identifier) @attribute.definition)

; Type references
(type_identifier) @class.reference
"""
)

GO_QUERIES = """
; Function declarations
(function_declaration
  name: (identifier) @function.definition)

; Method declarations
(method_declaration
  name: (field_identifier) @method.definition)

; Type declarations (struct, interface)
(type_declaration
  (type_spec
    name: (type_identifier) @class.definition))

; Variable declarations
(var_declaration
  (var_spec
    name: (identifier) @variable.definition))
(short_var_declaration
  left: (expression_list
    (identifier) @variable.definition))

; Const declarations
(const_declaration
  (const_spec
    name: (identifier) @variable.definition))

; Parameters
(parameter_declaration
  name: (identifier) @parameter.definition)

; Import declarations
(import_spec
  path: (interpreted_string_literal) @import.definition)

; Function calls
(call_expression
  function: (identifier) @function.reference)
(call_expression
  function: (selector_expression
    field: (field_identifier) @method.reference))

; Field access
(selector_expression
  field: (field_identifier) @attribute.reference)

; Identifier references
(identifier) @identifier.reference
(type_identifier) @class.reference
"""

RUST_QUERIES = """
; Function definitions
(function_item
  name: (identifier) @function.definition)

; Struct definitions
(struct_item
  name: (type_identifier) @class.definition)

; Enum definitions
(enum_item
  name: (type_identifier) @class.definition)

; Trait definitions
(trait_item
  name: (type_identifier) @class.definition)

; Impl blocks
(impl_item
  type: (type_identifier) @class.reference)

; Method definitions in impl
(impl_item
  body: (declaration_list
    (function_item
      name: (identifier) @method.definition)))

; Let bindings
(let_declaration
  pattern: (identifier) @variable.definition)

; Const declarations
(const_item
  name: (identifier) @variable.definition)

; Static declarations
(static_item
  name: (identifier) @variable.definition)

; Parameters
(parameter
  pattern: (identifier) @parameter.definition)

; Use statements
(use_declaration
  argument: (scoped_identifier
    name: (identifier) @import.definition))
(use_declaration
  argument: (identifier) @import.definition)

; Function calls
(call_expression
  function: (identifier) @function.reference)
(call_expression
  function: (scoped_identifier
    name: (identifier) @function.reference))

; Method calls
(call_expression
  function: (field_expression
    field: (field_identifier) @method.reference))

; Field access
(field_expression
  field: (field_identifier) @attribute.reference)

; Identifier references
(identifier) @identifier.reference
(type_identifier) @class.reference
"""

JAVA_QUERIES = """
; Class declarations
(class_declaration
  name: (identifier) @class.definition)

; Interface declarations
(interface_declaration
  name: (identifier) @class.definition)

; Enum declarations
(enum_declaration
  name: (identifier) @class.definition)

; Method declarations
(method_declaration
  name: (identifier) @method.definition)

; Constructor declarations
(constructor_declaration
  name: (identifier) @method.definition)

; Field declarations
(field_declaration
  declarator: (variable_declarator
    name: (identifier) @attribute.definition))

; Local variable declarations
(local_variable_declaration
  declarator: (variable_declarator
    name: (identifier) @variable.definition))

; Parameters
(formal_parameter
  name: (identifier) @parameter.definition)

; Import declarations
(import_declaration
  (scoped_identifier) @import.definition)

; Method invocations
(method_invocation
  name: (identifier) @method.reference)

; Field access
(field_access
  field: (identifier) @attribute.reference)

; Identifier references
(identifier) @identifier.reference
(type_identifier) @class.reference
"""

C_QUERIES = """
; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @function.definition))

; Function declarations
(declaration
  declarator: (function_declarator
    declarator: (identifier) @function.definition))

; Struct definitions
(struct_specifier
  name: (type_identifier) @class.definition)

; Enum definitions
(enum_specifier
  name: (type_identifier) @class.definition)

; Typedef
(type_definition
  declarator: (type_identifier) @class.definition)

; Variable declarations
(declaration
  declarator: (init_declarator
    declarator: (identifier) @variable.definition))

; Parameters
(parameter_declaration
  declarator: (identifier) @parameter.definition)

; Include directives
(preproc_include
  path: (_) @import.definition)

; Function calls
(call_expression
  function: (identifier) @function.reference)

; Field access
(field_expression
  field: (field_identifier) @attribute.reference)

; Identifier references
(identifier) @identifier.reference
(type_identifier) @class.reference
"""

CPP_QUERIES = (
    C_QUERIES
    + """
; Class definitions
(class_specifier
  name: (type_identifier) @class.definition)

; Namespace definitions
(namespace_definition
  name: (identifier) @class.definition)

; Method definitions
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @method.definition)))

; Using declarations
(using_declaration
  (qualified_identifier) @import.definition)
"""
)

CSHARP_QUERIES = """
; Class declarations
(class_declaration
  name: (identifier) @class.definition)

; Interface declarations
(interface_declaration
  name: (identifier) @class.definition)

; Struct declarations
(struct_declaration
  name: (identifier) @class.definition)

; Enum declarations
(enum_declaration
  name: (identifier) @class.definition)

; Method declarations
(method_declaration
  name: (identifier) @method.definition)

; Constructor declarations
(constructor_declaration
  name: (identifier) @method.definition)

; Property declarations
(property_declaration
  name: (identifier) @attribute.definition)

; Field declarations
(field_declaration
  (variable_declaration
    (variable_declarator
      (identifier) @attribute.definition)))

; Local variable declarations
(local_declaration_statement
  (variable_declaration
    (variable_declarator
      (identifier) @variable.definition)))

; Parameters
(parameter
  name: (identifier) @parameter.definition)

; Using directives
(using_directive
  (qualified_name) @import.definition)
(using_directive
  (identifier) @import.definition)

; Method invocations
(invocation_expression
  function: (member_access_expression
    name: (identifier) @method.reference))

; Member access
(member_access_expression
  name: (identifier) @attribute.reference)

; Identifier references
(identifier) @identifier.reference
"""

RUBY_QUERIES = """
; Class definitions
(class
  name: (constant) @class.definition)

; Module definitions
(module
  name: (constant) @class.definition)

; Method definitions
(method
  name: (identifier) @method.definition)

; Singleton method definitions
(singleton_method
  name: (identifier) @method.definition)

; Assignment
(assignment
  left: (identifier) @variable.definition)

; Instance variable assignment
(assignment
  left: (instance_variable) @attribute.definition)

; Parameters
(method_parameters
  (identifier) @parameter.definition)
(block_parameters
  (identifier) @parameter.definition)

; Require statements
(call
  method: (identifier) @_require
  arguments: (argument_list
    (string
      (string_content) @import.definition))
  (#eq? @_require "require"))

; Method calls
(call
  method: (identifier) @method.reference)

; Identifier references
(identifier) @identifier.reference
(constant) @class.reference
"""

PHP_QUERIES = """
; Class declarations
(class_declaration
  name: (name) @class.definition)

; Interface declarations
(interface_declaration
  name: (name) @class.definition)

; Trait declarations
(trait_declaration
  name: (name) @class.definition)

; Method declarations
(method_declaration
  name: (name) @method.definition)

; Function definitions
(function_definition
  name: (name) @function.definition)

; Property declarations
(property_declaration
  (property_element
    (variable_name) @attribute.definition))

; Variable declarations
(simple_parameter
  name: (variable_name) @parameter.definition)

; Use statements
(namespace_use_clause
  (qualified_name) @import.definition)

; Method calls
(member_call_expression
  name: (name) @method.reference)

; Function calls
(function_call_expression
  function: (name) @function.reference)

; Property access
(member_access_expression
  name: (name) @attribute.reference)

; Variable references
(variable_name) @identifier.reference
(name) @identifier.reference
"""

# Map language to query string
LANGUAGE_QUERIES: dict[str, str] = {
    "python": PYTHON_QUERIES,
    "javascript": JAVASCRIPT_QUERIES,
    "typescript": TYPESCRIPT_QUERIES,
    "tsx": TYPESCRIPT_QUERIES,
    "go": GO_QUERIES,
    "rust": RUST_QUERIES,
    "java": JAVA_QUERIES,
    "c": C_QUERIES,
    "cpp": CPP_QUERIES,
    "c_sharp": CSHARP_QUERIES,
    "ruby": RUBY_QUERIES,
    "php": PHP_QUERIES,
    # Languages without specific queries use generic fallback
}


class CodeAnalyzer:
    """Tree-sitter based code analyzer for extracting symbols."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self._parsers: dict[str, object] = {}
        self._queries: dict[str, object] = {}
        self._languages: dict[str, object] = {}

    def _get_parser(self, language: str) -> "Tree | None":
        """Get or create a parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(language)
            self._parsers[language] = parser
            return parser
        except (ImportError, LookupError) as e:
            logger.warning("Failed to get parser for %s: %s", language, e)
            return None

    def _get_language(self, language: str) -> object | None:
        """Get tree-sitter language object."""
        if language in self._languages:
            return self._languages[language]

        try:
            from tree_sitter_language_pack import get_language

            lang = get_language(language)
            self._languages[language] = lang
            return lang
        except (ImportError, LookupError) as e:
            logger.warning("Failed to get language for %s: %s", language, e)
            return None

    def _get_query(self, language: str) -> object | None:
        """Get or create query for the given language."""
        if language in self._queries:
            return self._queries[language]

        query_str = LANGUAGE_QUERIES.get(language)
        if not query_str:
            return None

        lang = self._get_language(language)
        if not lang:
            return None

        try:
            from tree_sitter import Query

            # Filter out invalid patterns for this language
            # Some patterns may not work for all languages
            query = Query(lang, query_str)
            self._queries[language] = query
            return query
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to create query for %s: %s", language, e)
            # Try with a simpler fallback query
            try:
                from tree_sitter import Query

                fallback_query = "(identifier) @identifier.reference"
                query = Query(lang, fallback_query)
                self._queries[language] = query
                return query
            except Exception:  # noqa: BLE001
                return None

    def _get_source_line(self, content: str, line: int) -> str:
        """Get a specific line from the source content (1-based)."""
        lines = content.split("\n")
        if 0 < line <= len(lines):
            return lines[line - 1].strip()
        return ""

    def _find_parent_scope(self, node: "Node", source: bytes) -> tuple[str | None, str | None]:
        """Find the enclosing class/function for a node."""
        parent = node.parent
        while parent:
            node_type = parent.type
            # Class-like definitions
            if node_type in (
                "class_definition",
                "class_declaration",
                "class_specifier",
                "struct_specifier",
                "interface_declaration",
                "trait_declaration",
                "module",  # Ruby
                "impl_item",  # Rust
            ):
                # Find the name child
                for child in parent.children:
                    if child.type in ("identifier", "type_identifier", "constant", "name"):
                        return (child.text.decode("utf-8"), "class")
            # Function-like definitions
            if node_type in (
                "function_definition",
                "function_declaration",
                "method_declaration",
                "method_definition",
                "method",
                "function_item",
            ):
                for child in parent.children:
                    if child.type in ("identifier", "property_identifier", "name"):
                        return (child.text.decode("utf-8"), "function")
                # Handle declarator pattern (C/C++)
                for child in parent.children:
                    if child.type in ("function_declarator", "declarator"):
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                return (subchild.text.decode("utf-8"), "function")
            parent = parent.parent
        return (None, None)

    def extract_symbols(self, content: str, language: str) -> list[SymbolInfo]:
        """Extract symbols from source code.

        Args:
            content: Source code content
            language: Tree-sitter language name

        Returns:
            List of extracted symbols
        """
        if not has_tree_sitter_support(language):
            logger.debug("No tree-sitter support for %s", language)
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        source_bytes = content.encode("utf-8")
        try:
            tree = parser.parse(source_bytes)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to parse content as %s: %s", language, e)
            return []

        query = self._get_query(language)
        if not query:
            # Fallback: extract all identifiers generically
            return self._extract_generic_identifiers(tree.root_node, content)

        symbols: list[SymbolInfo] = []
        seen: set[tuple[str, int, int]] = set()  # Dedupe by (name, line, column)

        try:
            # tree-sitter 0.24+ uses QueryCursor for captures
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            # New API returns dict[str, list[Node]]
            captures: list[tuple[object, str]] = []
            for capture_name, nodes in captures_dict.items():
                for node in nodes:
                    captures.append((node, capture_name))
        except Exception as e:  # noqa: BLE001
            logger.warning("Query failed for %s: %s", language, e)
            return self._extract_generic_identifiers(tree.root_node, content)

        for node, capture_name in captures:
            # Parse capture name: kind.category (e.g., "class.definition", "method.reference")
            parts = capture_name.split(".")
            if len(parts) != 2:
                continue

            kind, category = parts

            # Skip generic identifier captures if we have more specific ones
            if kind == "identifier":
                continue

            name = node.text.decode("utf-8")
            line = node.start_point[0] + 1  # Convert to 1-based
            column = node.start_point[1]
            end_line = node.end_point[0] + 1
            end_column = node.end_point[1]

            # Dedupe
            key = (name, line, column)
            if key in seen:
                continue
            seen.add(key)

            # Get context
            context = self._get_source_line(content, line)

            # Find parent scope
            parent_name, parent_kind = self._find_parent_scope(node, source_bytes)

            # For attribute access, try to get the object name
            if kind == "attribute" and category == "reference":
                parent = node.parent
                if parent and parent.type in ("attribute", "member_expression", "field_expression"):
                    obj_node = parent.child_by_field_name("object")
                    if obj_node is None:
                        # Try first child
                        for child in parent.children:
                            if child.type in ("identifier", "self", "this"):
                                parent_name = child.text.decode("utf-8")
                                break
                    elif obj_node:
                        parent_name = obj_node.text.decode("utf-8")

            symbols.append(
                SymbolInfo(
                    name=name,
                    kind=kind,
                    category=category,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                    context=context,
                    parent_name=parent_name,
                    parent_kind=parent_kind,
                )
            )

        return symbols

    def _extract_generic_identifiers(self, root_node: "Node", content: str) -> list[SymbolInfo]:
        """Fallback: extract all identifiers from the AST."""
        symbols: list[SymbolInfo] = []
        seen: set[tuple[str, int, int]] = set()

        def visit(node: "Node") -> None:
            if node.type in ("identifier", "type_identifier", "property_identifier"):
                name = node.text.decode("utf-8")
                line = node.start_point[0] + 1
                column = node.start_point[1]

                key = (name, line, column)
                if key not in seen:
                    seen.add(key)
                    symbols.append(
                        SymbolInfo(
                            name=name,
                            kind="identifier",
                            category="reference",
                            line=line,
                            column=column,
                            end_line=node.end_point[0] + 1,
                            end_column=node.end_point[1],
                            context=self._get_source_line(content, line),
                        )
                    )

            for child in node.children:
                visit(child)

        visit(root_node)
        return symbols


# Module-level analyzer instance (singleton)
_analyzer: CodeAnalyzer | None = None


def get_analyzer() -> CodeAnalyzer:
    """Get or create the module-level analyzer instance."""
    global _analyzer  # noqa: PLW0603
    if _analyzer is None:
        _analyzer = CodeAnalyzer()
    return _analyzer


def extract_symbols_from_file(content: str, file_path: str) -> list[SymbolInfo]:
    """Extract symbols from a file's content.

    Args:
        content: File content
        file_path: Path to the file (used for language detection)

    Returns:
        List of extracted symbols
    """
    language = detect_language(file_path)
    if not language:
        logger.debug("Unknown language for %s", file_path)
        return []

    if should_skip_language(language):
        logger.debug("Skipping data format %s for %s", language, file_path)
        return []

    analyzer = get_analyzer()
    return analyzer.extract_symbols(content, language)
