import sys
import subprocess
import threading
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, 
    QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QLabel, 
    QFrame, QListWidget, QMessageBox, QLineEdit
)
from PyQt6.QtCore import Qt, QFileInfo, QDir
from PyQt6.QtGui import QFont, QTextCursor, QPainter, QColor


class ModernIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern C++ IDE")
        self.setGeometry(100, 100, 1200, 800)

        # Prompt user to select a project folder
        self.project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not self.project_dir:
            QMessageBox.critical(self, "Error", "No folder selected. Exiting.")
            sys.exit()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.main_layout = QHBoxLayout(self.central_widget)
        self.sidebar = InteractiveSidebar(self)
        self.editor = CodeEditor(self)
        self.console = Console(self.editor, self.project_dir)  # Pass editor and project directory to console

        # Adding widgets to the main layout
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.editor)
        self.main_layout.addWidget(self.console)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Confirm Exit', 
            "Are you sure you want to exit?", 
            QMessageBox.StandardButton.Yes | 
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


class InteractiveSidebar(QFrame):
    def __init__(self, master):
        super().__init__(master)
        self.setFixedWidth(200)
        
        # Sidebar layout
        layout = QVBoxLayout()
        
        # File operations
        self.file_list = QListWidget()
        layout.addWidget(QLabel("Files"))
        layout.addWidget(self.file_list)

        # Buttons for file operations
        add_file_button = QPushButton("Add New File")
        add_file_button.clicked.connect(self.add_file)

        add_existing_file_button = QPushButton("Add Existing File")
        add_existing_file_button.clicked.connect(self.add_existing_file)
        
        layout.addWidget(add_file_button)
        layout.addWidget(add_existing_file_button)

        self.setLayout(layout)

    def add_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Create New File", self.parent().project_dir, "C++ Files (*.cpp);;All Files (*)")
        if file_name:
            self.file_list.addItem(file_name)

    def add_existing_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Add Existing File", self.parent().project_dir, "C++ Files (*.cpp);;All Files (*)")
        if file_name:
            self.file_list.addItem(file_name)


class CodeEditor(QTextEdit):
    def __init__(self, master):
        super().__init__(master)
        
        # Editor properties
        self.font_size = 12
        self.setFont(QFont("Courier", self.font_size, QFont.Weight.Bold))
        self.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")

    def keyPressEvent(self, event):
        """Handle key press events for code completion."""
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_ParenLeft, Qt.Key.Key_BracketLeft):
            closing_char = ')' if event.key() == Qt.Key.Key_ParenLeft else ']'
            cursor = self.textCursor()
            cursor.insertText(closing_char)

    def wheelEvent(self, event):
        """Zoom in/out with Ctrl + Mouse Wheel."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.font_size += 1
            else:
                self.font_size -= 1
            self.setFont(QFont("Courier", self.font_size, QFont.Weight.Bold))
        else:
            super().wheelEvent(event)


class LineNumberArea(QFrame):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), Qt.GlobalColor.lightGray)

        line_count = len(self.editor.toPlainText().split('\n'))
        for line_number in range(1, line_count + 1):
            painter.drawText(0, line_number * 20 - 10, str(line_number))

    def update_line_numbers(self):
        self.update()


class Console(QFrame):
    def __init__(self, editor, project_dir):
        super().__init__()
        self.project_dir = project_dir
        self.font_size = 12
        self.setFont(QFont("Courier", self.font_size, QFont.Weight.Bold))
        self.setStyleSheet("background-color: #282c34; color: #ffffff;")
        
        layout = QVBoxLayout()
        
        # Console output area
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        
        # Command input area
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter a command...")
        self.command_input.returnPressed.connect(self.execute_custom_command)

        # Labels and run buttons
        layout.addWidget(QLabel("Console Output"))
        layout.addWidget(self.console_output)
        layout.addWidget(self.command_input)

        run_button = QPushButton("Run Code")
        run_button.clicked.connect(lambda: self.run_code(editor))

        layout.addWidget(run_button)
        
        self.setLayout(layout)

    def insert_text(self, text):
        cursor = QTextCursor(self.console_output.textCursor())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n")
        self.console_output.setTextCursor(cursor)

    def run_code(self, editor):
        """Compile and run code from the editor in the project directory."""
        code = editor.toPlainText()
        cpp_path = os.path.join(self.project_dir, 'temp.cpp')
        exe_path = os.path.join(self.project_dir, 'temp.exe')

        with open(cpp_path, 'w') as f:
            f.write(code)

        command = f"g++ \"{cpp_path}\" -o \"{exe_path}\" && \"{exe_path}\"" if sys.platform == 'win32' else f"g++ \"{cpp_path}\" -o \"{exe_path}\" && ./temp"
        
        threading.Thread(target=self.execute_command, args=(command,), daemon=True).start()

    def execute_command(self, command):
        """Execute a shell command and capture its output."""
        process = subprocess.Popen(
            command, shell=True, cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        out, err = process.communicate()
        
        output_text = out.decode() + err.decode()
        
        if output_text:
            self.insert_text(output_text)

    def execute_custom_command(self):
        command = self.command_input.text()
        self.command_input.clear()
        if command:
            threading.Thread(target=self.execute_command, args=(command,), daemon=True).start()


    def zoom_in(self):
        """Increase font size in the console."""
        self.font_size += 1
        self.setFont(QFont("Courier", self.font_size))
        self.console_output.setFont(QFont("Courier", self.font_size))

    def zoom_out(self):
        """Decrease font size in the console."""
        if self.font_size > 8:  # Set a minimum font size
            self.font_size -= 1
            self.setFont(QFont("Courier", self.font_size))
            self.console_output.setFont(QFont("Courier", self.font_size))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ide = ModernIDE()
    ide.show()
    sys.exit(app.exec())
