"""
app/dialogs/scale_dialog.py

Diálogo para calibração de escala µm/pixel.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QPushButton, QLabel, QGroupBox, QRadioButton
)
from PyQt6.QtCore import Qt
from acquisition.scale_manager import ScaleManager


class ScaleDialog(QDialog):
    """
    Diálogo para calibrar a escala µm/pixel.
    Suporta dois métodos: barra de escala ou valor direto.
    """

    def __init__(self, scale_manager: ScaleManager, parent=None):
        super().__init__(parent)
        self.scale_manager = scale_manager
        self.setWindowTitle("Calibração de Escala")
        self.setMinimumWidth(400)
        self.setModal(True)
        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("⚖️  Calibração de Escala (µm/pixel)")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00b4d8;")
        layout.addWidget(title)

        info = QLabel(
            "⚠️  A escala é OBRIGATÓRIA para medições em µm reais.\n"
            "Sem escala, resultados ficam em pixels (sem valor metrológico)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #f4a261; font-size: 11px; padding: 4px;")
        layout.addWidget(info)

        # --- Método 1: Barra de escala ---
        self.radio_bar = QRadioButton("Método 1: Medir barra de escala na imagem")
        self.radio_bar.setChecked(True)
        layout.addWidget(self.radio_bar)

        self.bar_group = QGroupBox()
        bar_form = QFormLayout(self.bar_group)

        self.bar_px_spin = QDoubleSpinBox()
        self.bar_px_spin.setRange(1, 50000)
        self.bar_px_spin.setValue(100)
        self.bar_px_spin.setSuffix(" px")
        self.bar_px_spin.setDecimals(1)
        bar_form.addRow("Comprimento da barra (pixels):", self.bar_px_spin)

        self.bar_um_spin = QDoubleSpinBox()
        self.bar_um_spin.setRange(0.001, 100000)
        self.bar_um_spin.setValue(100)
        self.bar_um_spin.setSuffix(" µm")
        self.bar_um_spin.setDecimals(3)
        bar_form.addRow("Comprimento real da barra (µm):", self.bar_um_spin)

        self.preview_label = QLabel("Escala calculada: — µm/px")
        self.preview_label.setStyleSheet("color: #06d6a0; font-weight: bold;")
        bar_form.addRow("", self.preview_label)

        # Atualiza preview em tempo real
        self.bar_px_spin.valueChanged.connect(self._update_preview)
        self.bar_um_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

        layout.addWidget(self.bar_group)

        # --- Método 2: Valor direto ---
        self.radio_direct = QRadioButton("Método 2: Inserir valor direto (µm/pixel)")
        layout.addWidget(self.radio_direct)

        self.direct_group = QGroupBox()
        direct_form = QFormLayout(self.direct_group)

        self.direct_spin = QDoubleSpinBox()
        self.direct_spin.setRange(0.0001, 1000)
        self.direct_spin.setValue(0.5)
        self.direct_spin.setSuffix(" µm/px")
        self.direct_spin.setDecimals(4)
        direct_form.addRow("Escala (µm/pixel):", self.direct_spin)
        self.direct_group.setEnabled(False)

        layout.addWidget(self.direct_group)

        # Alterna habilitação dos grupos
        self.radio_bar.toggled.connect(self.bar_group.setEnabled)
        self.radio_direct.toggled.connect(self.direct_group.setEnabled)

        # Botões
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_ok = QPushButton("✔  Aplicar Escala")
        btn_ok.clicked.connect(self._apply_scale)
        btn_layout.addWidget(btn_ok)

        layout.addLayout(btn_layout)

    def _update_preview(self):
        """Mostra a escala calculada em tempo real."""
        try:
            scale = self.bar_um_spin.value() / self.bar_px_spin.value()
            self.preview_label.setText(f"Escala calculada: {scale:.4f} µm/px")
        except ZeroDivisionError:
            self.preview_label.setText("Escala calculada: —")

    def _apply_scale(self):
        """Calibra o ScaleManager com o método selecionado."""
        if self.radio_bar.isChecked():
            self.scale_manager.calibrate_from_bar(
                bar_length_px=self.bar_px_spin.value(),
                bar_length_um=self.bar_um_spin.value()
            )
        else:
            self.scale_manager.calibrate_from_value(self.direct_spin.value())
        self.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog { background-color: #0f0f23; }
            QGroupBox {
                color: #a0a0c0;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                margin-top: 4px;
                padding: 8px;
            }
            QLabel { color: #c0c0e0; font-size: 12px; }
            QRadioButton { color: #c0c0e0; font-size: 12px; }
            QDoubleSpinBox {
                background-color: #1a1a2e;
                color: #e0e0ff;
                border: 1px solid #2a2a4a;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
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
        """)
