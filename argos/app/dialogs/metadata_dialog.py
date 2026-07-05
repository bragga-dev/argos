"""
app/dialogs/metadata_dialog.py

Diálogo para o usuário preencher os metadados da amostra.
Aparece após carregar uma imagem.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QLabel, QTextEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from acquisition.metadata import ImageMetadata


class MetadataDialog(QDialog):
    """
    Janela de diálogo para preenchimento de metadados metalográficos.
    """

    ETCHANTS = [
        "Não informado",
        "Nital 2%",
        "Nital 4%",
        "Nital 10%",
        "Reagente de Marble",
        "FeCl3 + HCl",
        "Vilella",
        "Klemm I",
        "Beraha",
        "Outro",
    ]

    STANDARDS = [
        "Não definida",
        "ASTM E112 — Tamanho de grão",
        "ASTM E45 — Inclusões não metálicas",
        "ISO 643 — Aço",
    ]

    def __init__(self, metadata: ImageMetadata, parent=None):
        super().__init__(parent)
        self.metadata = metadata
        self.setWindowTitle("Metadados da Amostra")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Cabeçalho
        header = QLabel(f"📋  {self.metadata.file_name}")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00b4d8;")
        layout.addWidget(header)

        # Grupo: Condições de aquisição
        acq_group = QGroupBox("Condições de Aquisição")
        acq_form = QFormLayout(acq_group)

        self.magnification_spin = QSpinBox()
        self.magnification_spin.setRange(10, 5000)
        self.magnification_spin.setValue(100)
        self.magnification_spin.setSuffix("x")
        acq_form.addRow("Aumento:", self.magnification_spin)

        self.etchant_combo = QComboBox()
        self.etchant_combo.addItems(self.ETCHANTS)
        acq_form.addRow("Ataque químico:", self.etchant_combo)

        self.etching_time_spin = QDoubleSpinBox()
        self.etching_time_spin.setRange(0, 3600)
        self.etching_time_spin.setValue(0)
        self.etching_time_spin.setSuffix(" s")
        acq_form.addRow("Tempo de ataque:", self.etching_time_spin)

        layout.addWidget(acq_group)

        # Grupo: Material
        mat_group = QGroupBox("Material e Norma")
        mat_form = QFormLayout(mat_group)

        self.material_edit = QLineEdit()
        self.material_edit.setPlaceholderText("Ex: AISI 1045, Fe-C 0.8%, Alumínio 6061")
        mat_form.addRow("Material:", self.material_edit)

        self.heat_treatment_edit = QLineEdit()
        self.heat_treatment_edit.setPlaceholderText("Ex: Normalizado, Temperado, Recozido")
        mat_form.addRow("Tratamento térmico:", self.heat_treatment_edit)

        self.standard_combo = QComboBox()
        self.standard_combo.addItems(self.STANDARDS)
        mat_form.addRow("Norma a aplicar:", self.standard_combo)

        layout.addWidget(mat_group)

        # Grupo: Operador
        op_group = QGroupBox("Rastreabilidade")
        op_form = QFormLayout(op_group)

        self.operator_edit = QLineEdit()
        self.operator_edit.setPlaceholderText("Nome do operador")
        op_form.addRow("Operador:", self.operator_edit)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Observações livres sobre a amostra...")
        self.notes_edit.setMaximumHeight(80)
        op_form.addRow("Observações:", self.notes_edit)

        layout.addWidget(op_group)

        # Botões
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("✔  Salvar Metadados")
        self.btn_ok.clicked.connect(self._save_and_accept)
        self.btn_ok.setDefault(True)
        btn_layout.addWidget(self.btn_ok)

        layout.addLayout(btn_layout)

    def _save_and_accept(self):
        """Salva os valores preenchidos no objeto metadata e fecha o diálogo."""
        self.metadata.magnification = self.magnification_spin.value()
        self.metadata.etchant = self.etchant_combo.currentText()
        self.metadata.etching_time_s = self.etching_time_spin.value()
        self.metadata.material = self.material_edit.text().strip() or None
        self.metadata.heat_treatment = self.heat_treatment_edit.text().strip() or None

        standard = self.standard_combo.currentText()
        self.metadata.standard = None if "Não definida" in standard else standard

        self.metadata.operator = self.operator_edit.text().strip() or None
        self.metadata.notes = self.notes_edit.toPlainText().strip() or None

        self.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0f0f23;
            }
            QGroupBox {
                color: #a0a0c0;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #6c9bcf;
            }
            QLabel { color: #c0c0e0; font-size: 12px; }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1a1a2e;
                color: #e0e0ff;
                border: 1px solid #2a2a4a;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #00b4d8;
            }
            QPushButton {
                background-color: #1a1a3e;
                color: #c0c0e0;
                border: 1px solid #2a2a4a;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2a2a5e; }
            QPushButton[default=true] {
                background-color: #00526e;
                color: white;
                border-color: #00b4d8;
            }
            QPushButton[default=true]:hover { background-color: #00748a; }
        """)
