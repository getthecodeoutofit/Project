# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QAction, QFileDialog, QTabWidget
from editor.code_editor import CodeEditor
from console.terminal import Terminal
from project_manager.project_explorer import ProjectExplorer
from settings.preferences import load_preferences, save_preferences
from PyQt5.QtGui import QIcon

class MyIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Advanced IDE")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load user preferences
        self.preferences = load_preferences()

        # Central Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        
        # Splitter and Tab Widget
        splitter = QSplitter()
        self.tabs = QTabWidget()
        splitter.addWidget(ProjectExplorer(self))
        splitter.addWidget(self.tabs)
        splitter.addWidget(Terminal())
        layout.addWidget(splitter)
        
        # New tab for Code Editor
        self.new_code_editor()

        # Create actions for menu bar
        self.create_actions()

    def create_actions(self):
        # File actions
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_current_file)
        
        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.save_as_file)
        
        new_tab_action = QAction("New Tab", self)
        new_tab_action.triggered.connect(self.new_code_editor)
        
        run_action = QAction("Run Code", self)
        run_action.triggered.connect(self.run_code)

        self.menuBar().addAction(save_action)
        self.menuBar().addAction(save_as_action)
        self.menuBar().addAction(new_tab_action)
        self.menuBar().addAction(run_action)

    def new_code_editor(self):
        editor = CodeEditor(self.preferences)
        self.tabs.addTab(editor, "Untitled")
        self.tabs.setCurrentWidget(editor)

    def save_current_file(self):
        editor = self.tabs.currentWidget()
        if isinstance(editor, CodeEditor):
            editor.save_file()

    def save_as_file(self):
        editor = self.tabs.currentWidget()
        if isinstance(editor, CodeEditor):
            editor.save_as_file()

    def run_code(self):
        editor = self.tabs.currentWidget()
        if isinstance(editor, CodeEditor):
            editor.run_code_in_terminal(self.findChild(Terminal))

    def closeEvent(self, event):
        # Save user preferences on exit
        self.preferences["window_size"] = self.size()
        self.preferences["window_position"] = self.pos()
        save_preferences(self.preferences)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyIDE()
    window.show()
    sys.exit(app.exec())
