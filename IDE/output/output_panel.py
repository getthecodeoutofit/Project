# output/output_panel.py
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtGui import QColor, QFont

class OutputPanel(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        font = QFont("Courier", 10)
        self.setFont(font)
        self.setPlaceholderText("Execution Output...")

    def display_output(self, output):
        self.clear()
        self.setPlainText(output)
