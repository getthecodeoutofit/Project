# project_manager/project_explorer.py

from PyQt5.QtWidgets import QTreeView, QFileSystemModel, QVBoxLayout, QWidget

class ProjectExplorer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        layout.addWidget(self.file_tree)
