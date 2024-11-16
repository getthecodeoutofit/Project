# settings/settings_panel.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton

class SettingsPanel(QDialog):
    def __init__(self, preferences):
        super().__init__()
        self.preferences = preferences
        self.setWindowTitle("Settings")
        self.resize(300, 200)

        layout = QVBoxLayout()

        # Theme Selection
        layout.addWidget(QLabel("Select Theme"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(preferences.get("theme", "Light"))
        layout.addWidget(self.theme_combo)
        
        # Font Size
        layout.addWidget(QLabel("Font Size"))
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems([str(i) for i in range(8, 20)])
        self.font_size_combo.setCurrentText(str(preferences.get("font_size", 12)))
        layout.addWidget(self.font_size_combo)

        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        self.setLayout(layout)

    def save_settings(self):
        self.preferences["theme"] = self.theme_combo.currentText()
        self.preferences["font_size"] = int(self.font_size_combo.currentText())
        self.accept()
