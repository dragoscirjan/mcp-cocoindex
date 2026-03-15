"""Tests for the tree-sitter code analyzer."""

import pytest

from mcp_coco_index.analyzer import (
    CodeAnalyzer,
    SymbolInfo,
    detect_language,
    extract_symbols_from_file,
    has_tree_sitter_support,
    should_skip_language,
)


class TestLanguageDetection:
    """Tests for language detection functions."""

    def test_detect_python(self):
        """Test Python file detection."""
        assert detect_language("test.py") == "python"
        assert detect_language("module.pyw") == "python"
        assert detect_language("stubs.pyi") == "python"

    def test_detect_javascript(self):
        """Test JavaScript file detection."""
        assert detect_language("app.js") == "javascript"
        assert detect_language("component.jsx") == "javascript"
        assert detect_language("module.mjs") == "javascript"

    def test_detect_typescript(self):
        """Test TypeScript file detection."""
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "tsx"

    def test_detect_go(self):
        """Test Go file detection."""
        assert detect_language("main.go") == "go"

    def test_detect_rust(self):
        """Test Rust file detection."""
        assert detect_language("lib.rs") == "rust"

    def test_detect_java(self):
        """Test Java file detection."""
        assert detect_language("Main.java") == "java"

    def test_detect_c_cpp(self):
        """Test C/C++ file detection."""
        assert detect_language("main.c") == "c"
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("main.cc") == "cpp"
        assert detect_language("header.h") == "cpp"
        assert detect_language("header.hpp") == "cpp"

    def test_detect_csharp(self):
        """Test C# file detection."""
        assert detect_language("Program.cs") == "c_sharp"

    def test_detect_ruby(self):
        """Test Ruby file detection."""
        assert detect_language("app.rb") == "ruby"

    def test_detect_php(self):
        """Test PHP file detection."""
        assert detect_language("index.php") == "php"

    def test_detect_unknown(self):
        """Test unknown file extensions."""
        assert detect_language("file.xyz") is None
        assert detect_language("noextension") is None

    def test_detect_case_insensitive(self):
        """Test case-insensitive extension detection."""
        assert detect_language("file.PY") == "python"
        assert detect_language("file.Py") == "python"


class TestTreeSitterSupport:
    """Tests for tree-sitter support checks."""

    def test_supported_languages(self):
        """Test that main languages are supported."""
        assert has_tree_sitter_support("python") is True
        assert has_tree_sitter_support("javascript") is True
        assert has_tree_sitter_support("typescript") is True
        assert has_tree_sitter_support("go") is True
        assert has_tree_sitter_support("rust") is True
        assert has_tree_sitter_support("java") is True

    def test_unsupported_languages(self):
        """Test unsupported/unknown languages."""
        assert has_tree_sitter_support(None) is False
        assert has_tree_sitter_support("unknown") is False

    def test_skip_data_formats(self):
        """Test that data formats are skipped."""
        assert should_skip_language("json") is True
        assert should_skip_language("yaml") is True
        assert should_skip_language("toml") is True
        assert should_skip_language("xml") is True
        assert should_skip_language("html") is True
        assert should_skip_language("markdown") is True
        assert should_skip_language("sql") is True

    def test_dont_skip_programming_languages(self):
        """Test that programming languages are not skipped."""
        assert should_skip_language("python") is False
        assert should_skip_language("javascript") is False
        assert should_skip_language(None) is False


class TestCodeAnalyzer:
    """Tests for the CodeAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CodeAnalyzer instance."""
        return CodeAnalyzer()

    def test_extract_python_class(self, analyzer):
        """Test extracting Python class definition."""
        code = """
class MyClass:
    pass
"""
        symbols = analyzer.extract_symbols(code, "python")
        class_defs = [s for s in symbols if s.kind == "class" and s.category == "definition"]
        assert len(class_defs) >= 1
        assert any(s.name == "MyClass" for s in class_defs)

    def test_extract_python_function(self, analyzer):
        """Test extracting Python function definition."""
        code = """
def my_function(arg1, arg2):
    return arg1 + arg2
"""
        symbols = analyzer.extract_symbols(code, "python")
        func_defs = [s for s in symbols if s.kind == "function" and s.category == "definition"]
        assert len(func_defs) >= 1
        assert any(s.name == "my_function" for s in func_defs)

    def test_extract_python_method(self, analyzer):
        """Test extracting Python method definition."""
        code = """
class MyClass:
    def my_method(self):
        pass
"""
        symbols = analyzer.extract_symbols(code, "python")
        # Methods in Python are captured as functions or methods depending on query
        method_defs = [
            s
            for s in symbols
            if s.kind in ("method", "function")
            and s.category == "definition"
            and s.name == "my_method"
        ]
        assert len(method_defs) >= 1
        assert any(s.name == "my_method" for s in method_defs)

    def test_extract_python_variable(self, analyzer):
        """Test extracting Python variable assignment."""
        code = """
my_var = 42
"""
        symbols = analyzer.extract_symbols(code, "python")
        var_defs = [s for s in symbols if s.kind == "variable" and s.category == "definition"]
        assert len(var_defs) >= 1
        assert any(s.name == "my_var" for s in var_defs)

    def test_extract_python_function_call(self, analyzer):
        """Test extracting Python function call reference."""
        code = """
result = print("hello")
"""
        symbols = analyzer.extract_symbols(code, "python")
        func_refs = [s for s in symbols if s.kind == "function" and s.category == "reference"]
        assert len(func_refs) >= 1
        assert any(s.name == "print" for s in func_refs)

    def test_symbol_info_has_line_numbers(self, analyzer):
        """Test that symbols have correct line numbers."""
        code = """# Line 1
# Line 2
def foo():  # Line 3
    pass
"""
        symbols = analyzer.extract_symbols(code, "python")
        foo_def = next((s for s in symbols if s.name == "foo" and s.category == "definition"), None)
        assert foo_def is not None
        assert foo_def.line == 3

    def test_extract_javascript_function(self, analyzer):
        """Test extracting JavaScript function declaration."""
        code = """
function myFunction() {
    return 42;
}
"""
        symbols = analyzer.extract_symbols(code, "javascript")
        func_defs = [s for s in symbols if s.kind == "function" and s.category == "definition"]
        assert len(func_defs) >= 1
        assert any(s.name == "myFunction" for s in func_defs)

    def test_extract_javascript_class(self, analyzer):
        """Test extracting JavaScript class declaration."""
        code = """
class MyClass {
    constructor() {}
}
"""
        symbols = analyzer.extract_symbols(code, "javascript")
        class_defs = [s for s in symbols if s.kind == "class" and s.category == "definition"]
        assert len(class_defs) >= 1
        assert any(s.name == "MyClass" for s in class_defs)

    def test_extract_go_function(self, analyzer):
        """Test extracting Go function declaration."""
        code = """
package main

func MyFunction() int {
    return 42
}
"""
        symbols = analyzer.extract_symbols(code, "go")
        func_defs = [s for s in symbols if s.kind == "function" and s.category == "definition"]
        assert len(func_defs) >= 1
        assert any(s.name == "MyFunction" for s in func_defs)

    def test_extract_rust_function(self, analyzer):
        """Test extracting Rust function declaration."""
        code = """
fn my_function() -> i32 {
    42
}
"""
        symbols = analyzer.extract_symbols(code, "rust")
        func_defs = [s for s in symbols if s.kind == "function" and s.category == "definition"]
        assert len(func_defs) >= 1
        assert any(s.name == "my_function" for s in func_defs)

    def test_extract_empty_code(self, analyzer):
        """Test extracting from empty code."""
        symbols = analyzer.extract_symbols("", "python")
        assert symbols == []

    def test_extract_whitespace_only(self, analyzer):
        """Test extracting from whitespace-only code."""
        symbols = analyzer.extract_symbols("   \n\n   ", "python")
        assert symbols == []


class TestExtractSymbolsFromFile:
    """Tests for the extract_symbols_from_file function."""

    def test_extract_from_python_file(self):
        """Test extracting symbols with Python file path."""
        code = """
class Foo:
    def bar(self):
        pass
"""
        symbols = extract_symbols_from_file(code, "test.py")
        assert len(symbols) >= 2
        names = {s.name for s in symbols if s.category == "definition"}
        assert "Foo" in names
        assert "bar" in names

    def test_extract_from_javascript_file(self):
        """Test extracting symbols with JavaScript file path."""
        code = """
function hello() {
    console.log("Hello");
}
"""
        symbols = extract_symbols_from_file(code, "app.js")
        func_defs = [s for s in symbols if s.kind == "function" and s.category == "definition"]
        assert any(s.name == "hello" for s in func_defs)

    def test_skip_unknown_extension(self):
        """Test that unknown extensions return empty list."""
        symbols = extract_symbols_from_file("some content", "file.xyz")
        assert symbols == []

    def test_skip_json_file(self):
        """Test that JSON files are skipped."""
        symbols = extract_symbols_from_file('{"key": "value"}', "data.json")
        assert symbols == []

    def test_skip_yaml_file(self):
        """Test that YAML files are skipped."""
        symbols = extract_symbols_from_file("key: value", "config.yaml")
        assert symbols == []


class TestSymbolInfo:
    """Tests for SymbolInfo dataclass."""

    def test_symbol_info_creation(self):
        """Test creating a SymbolInfo instance."""
        symbol = SymbolInfo(
            name="test_func",
            kind="function",
            category="definition",
            line=10,
            column=0,
            end_line=15,
            end_column=0,
            context="def test_func():",
        )
        assert symbol.name == "test_func"
        assert symbol.kind == "function"
        assert symbol.category == "definition"
        assert symbol.line == 10
        assert symbol.column == 0
        assert symbol.parent_name is None

    def test_symbol_info_with_parent(self):
        """Test SymbolInfo with parent information."""
        symbol = SymbolInfo(
            name="my_method",
            kind="method",
            category="definition",
            line=5,
            column=4,
            end_line=5,
            end_column=13,
            context="def my_method(self):",
            parent_name="MyClass",
            parent_kind="class",
        )
        assert symbol.parent_name == "MyClass"
        assert symbol.parent_kind == "class"
