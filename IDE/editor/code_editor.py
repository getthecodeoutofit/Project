# editor/code_editor.py

from PyQt5.QtWidgets import QPlainTextEdit, QFileDialog
from PyQt5.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PyQt5.QtCore import Qt, QRegularExpression
import os
import tempfile
import subprocess

class CodeEditor(QPlainTextEdit):
    def __init__(self, preferences):
        super().__init__()
        self.preferences = preferences
        self.file_path = None  # Track the current file path
        self.apply_preferences()
        self.highlighter = PythonSyntaxHighlighter(self.document())

    def apply_preferences(self):
        font = QFont(self.preferences.get("font", "Courier"))
        font.setPointSize(self.preferences.get("font_size", 12))
        self.setFont(font)
        self.setStyleSheet(f"color: {self.preferences.get('font_color', '#000000')}")
        
    def save_file(self):
        if self.file_path is None:
            self.save_as_file()
        else:
            self.write_to_file(self.file_path)

    def save_as_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.file_path = file_path
            self.write_to_file(file_path)

    def write_to_file(self, path):
        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(self.toPlainText())
            print(f"File saved: {path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    def run_code_in_terminal(self, terminal):
        if terminal and self.file_path:
            command = f"python {self.file_path}" if self.file_path.endswith('.py') else f"gcc {self.file_path} -o output && ./output"
            terminal.run_command(command)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text):
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("blue"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = ["def", "class", "import", "from", "as", "if", "else", "elif", "return", "while", "for", "try", "except", "finally", "with", "assert", "lambda", "yield"]

        for word in keywords:
            expression = QRegularExpression(rf"\b{word}\b")
            match = expression.globalMatch(text)
            while match.hasNext():
                m = match.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), keyword_format)
