"""Dark theme (Qt style sheet) and matching matplotlib colours."""

BG, PANEL, PANEL2, BORDER = "#12161c", "#1a2029", "#202833", "#2c3644"
TEXT, MUTED, ACCENT, GOOD, WARN, BAD = "#d9e1ec", "#8794a7", "#3d9bff", "#3ecf8e", "#f0b429", "#ef5b5b"

QSS = f"""
* {{ font-family: 'Segoe UI', 'Inter', 'DejaVu Sans', sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QDialog, QWidget#root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {ACCENT}; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; background: {PANEL}; top: -1px; }}
QTabBar::tab {{ background: {BG}; border: 1px solid {BORDER}; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; border-bottom-color: {PANEL}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; background: #26303d; }}
QPushButton:pressed {{ background: #1b2430; }}
QPushButton:disabled {{ color: #566174; background: {PANEL}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton#primary:disabled {{ background: #27415f; color: #7f93ab; }}
QPushButton#danger {{ background: #7a2f35; border-color: {BAD}; }}
QPushButton#danger:disabled {{ background: {PANEL}; border-color: {BORDER}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 6px; selection-background-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QTableWidget, QTableView {{ background: {BG}; gridline-color: {BORDER}; border: 1px solid {BORDER}; alternate-background-color: {PANEL}; }}
QHeaderView::section {{ background: {PANEL2}; border: none; border-right: 1px solid {BORDER}; padding: 4px; color: {MUTED}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }} QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; }} QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QLabel#card_value {{ font-size: 17px; font-weight: 600; }}
QLabel#card_title {{ color: {MUTED}; font-size: 10px; text-transform: uppercase; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QMenuBar {{ background: {PANEL}; }} QMenuBar::item:selected {{ background: {PANEL2}; }}
QMenu {{ background: {PANEL2}; border: 1px solid {BORDER}; }} QMenu::item:selected {{ background: {ACCENT}; }}
QRadioButton, QCheckBox {{ spacing: 6px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {ACCENT}; }}
"""

MPL_RC = {"figure.facecolor": PANEL, "axes.facecolor": BG, "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
          "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "axes.titlecolor": TEXT,
          "grid.color": BORDER, "axes.grid": True, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
          "ytick.labelsize": 7, "legend.fontsize": 6, "legend.facecolor": PANEL2, "legend.edgecolor": BORDER}
