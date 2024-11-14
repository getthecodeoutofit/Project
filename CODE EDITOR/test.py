import sys
import subprocess
import threading
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, 
    QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QLabel, 
    QFrame, QTreeView, QMessageBox, QLineEdit, QCompleter
)
from PyQt6.QtCore import Qt, QDir, QRegularExpression
from PyQt6.QtGui import QFont, QTextCursor, QBrush, QColor, QSyntaxHighlighter, QTextCharFormat, QStandardItemModel, QStandardItem

class SyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_highlighting_rules()

    def init_highlighting_rules(self):
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QBrush(QColor(86, 156, 214)))
        keywords = ["int", "float", "return", "if", "else", "while", "for", "class", "public", "private", "void"]
        self.highlighting_rules += [(rf"\b{word}\b", keyword_format) for word in keywords]

        comment_format = QTextCharFormat()
        comment_format.setForeground(QBrush(QColor(87, 166, 74)))
        self.highlighting_rules.append((r"//[^\n]*", comment_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QBrush(QColor(214, 157, 133)))
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))

    def highlightBlock(self, text):
        for pattern, format_ in self.highlighting_rules:
            expression = QRegularExpression(pattern)
            match_iterator = expression.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format_)

class ModernIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern C++ IDE")
        self.setGeometry(100, 100, 1200, 800)

        self.project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not self.project_dir:
            QMessageBox.critical(self, "Error", "No folder selected. Exiting.")
            sys.exit()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.main_layout = QHBoxLayout(self.central_widget)
        self.sidebar = InteractiveSidebar(self, self.project_dir)  # Pass project_dir
        self.editor = CodeEditor(self)
        self.console = Console(self.editor, self.project_dir)  # Pass project_dir

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
    def __init__(self, master, project_dir):
        super().__init__(master)
        self.project_dir = project_dir
        self.setFixedWidth(200)
        
        layout = QVBoxLayout()
        self.file_tree = QTreeView()
        self.model = QStandardItemModel()
        self.file_tree.setModel(self.model)
        
        # Initialize project directory tree
        self.load_project_tree(self.model, self.project_dir)
        
        layout.addWidget(QLabel("Files"))
        layout.addWidget(self.file_tree)

        add_file_button = QPushButton("Add New File")
        add_file_button.clicked.connect(self.add_file)

        add_existing_file_button = QPushButton("Add Existing File")
        add_existing_file_button.clicked.connect(self.add_existing_file)
        
        layout.addWidget(add_file_button)
        layout.addWidget(add_existing_file_button)
        self.setLayout(layout)

    def load_project_tree(self, parent_item, path):
        for name in sorted(os.listdir(path)):
            full_path = os.path.join(path, name)
            item = QStandardItem(name)
            if os.path.isdir(full_path):
                item.setEditable(False)
                self.load_project_tree(item, full_path)
            parent_item.appendRow(item)

    def add_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Create New File", self.project_dir, "C++ Files (*.cpp);;All Files (*)")
        if file_name:
            # Create the new file on disk
            with open(file_name, 'w') as f:
                f.write("")  # Create an empty file

            # Add the file to the tree view
            item = QStandardItem(os.path.basename(file_name))
            self.model.appendRow(item)

    def add_existing_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Add Existing File", self.project_dir, "C++ Files (*.cpp);;All Files (*)")
        if file_name:
            # Add the existing file to the tree view
            item = QStandardItem(os.path.basename(file_name))
            self.model.appendRow(item)


class CodeEditor(QTextEdit):
    def __init__(self, master):
        super().__init__(master)
        
        self.font_size = 12
        self.setFont(QFont("Courier", self.font_size, QFont.Weight.Bold))
        self.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")
        self.highlighter = SyntaxHighlighter(self.document())
        
        # Auto-completion setup
        self.completer = QCompleter(["int", "float", "return", "if", "else", "while", "for", "class", "public", "private", "void"])
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_ParenLeft, Qt.Key.Key_BracketLeft):
            closing_char = ')' if event.key() == Qt.Key.Key_ParenLeft else ']'
            cursor = self.textCursor()
            cursor.insertText(closing_char)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.font_size += 1
            else:
                self.font_size -= 1
            self.setFont(QFont("Courier", self.font_size, QFont.Weight.Bold))
        else:
            super().wheelEvent(event)


class Console(QFrame):
    def __init__(self, editor, project_dir):
        super().__init__()
        self.project_dir = project_dir
        self.font_size = 12
        self.setFont(QFont("Courier", self.font_size, QFont.Weight.Bold))
        self.setStyleSheet("background-color: #282c34; color: #ffffff;")
        
        layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter a command...")
        self.command_input.returnPressed.connect(self.execute_custom_command)

        layout.addWidget(QLabel("Console Output"))
        layout.addWidget(self.console_output)
        layout.addWidget(self.command_input)

        run_button = QPushButton("Run Code")
        run_button.clicked.connect(lambda: self.run_code(editor))

        layout.addWidget(run_button)
        self.setLayout(layout)

        self.command_history = []
        self.command_index = -1

    def insert_text(self, text):
        cursor = QTextCursor(self.console_output.textCursor())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n")
        self.console_output.setTextCursor(cursor)

    def run_code(self, editor):
        code = editor.toPlainText()
        cpp_path = os.path.join(self.project_dir, 'temp.cpp')
        exe_path = os.path.join(self.project_dir, 'temp.exe')

        with open(cpp_path, 'w') as f:
            f.write(code)

        command = f"g++ \"{cpp_path}\" -o \"{exe_path}\" && \"{exe_path}\"" if sys.platform == 'win32' else f"g++ \"{cpp_path}\" -o \"{exe_path}\" && ./temp"
        
        threading.Thread(target=self.execute_command, args=(command,), daemon=True).start()

    def execute_command(self, command):
        process = subprocess.Popen(command, shell=True, cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        output_text = out.decode() + err.decode()
        if output_text:
            self.insert_text(output_text)

    def execute_custom_command(self):
        command = self.command_input.text()
        self.command_input.clear()
        if command:
            self.command_history.append(command)
            threading.Thread(target=self.execute_command, args=(command,), daemon=True).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ide = ModernIDE()
    ide.show()
    sys.exit(app.exec())
