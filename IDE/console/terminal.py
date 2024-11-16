# console/terminal.py

from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtCore import Qt
import subprocess

class Terminal(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(False)
        self.appendPlainText(">> ")

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Return:
            command = self.toPlainText().split("\n")[-1].strip(">> ")
            self.run_command(command)

    def run_command(self, command):
        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            self.appendPlainText(stdout.decode() if stdout else stderr.decode())
            self.appendPlainText(">> ")
        except Exception as e:
            self.appendPlainText(str(e))
