"""
app/main_window.py

Janela principal do ARGOS.
Conecta todos os módulos: aquisição → pré-processamento → visualização.

MUDANÇA DE UI (v2):
    Cada filtro agora é um par CHECKBOX + SLIDER, em vez de botão "Aplicar":
        • Checkbox desmarcado → filtro desligado, slider desabilitado
        • Checkbox marcado → filtro ligado, valor do slider é aplicado
        • Mover o slider (com checkbox marcado) atualiza o resultado
          em tempo real

    Toda mudança (toggle ou slider) chama self._on_state_changed(), que:
        1. Atualiza o estado correspondente no ImageProcessor
        2. Chama processor.rebuild() — recalcula do zero a partir do original
        3. Atualiza o viewer com o novo resultado

Layout:
    ┌─────────────────────────────────────────────┐
    │  Toolbar (Abrir, Escala, Metadados)          │
    ├──────────────────────┬──────────────────────┤
    │   Painel Esquerdo    │   Painel Direito      │
    │   (checkbox+slider   │   (Original | Atual)  │
    │    por filtro)       │                       │
    ├──────────────────────┴──────────────────────┤
    │  Status Bar (escala, dimensões, operações)   │
    └─────────────────────────────────────────────┘
"""

from typing import Callable, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QStatusBar, QLabel, QSlider,
    QGroupBox, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QFrame, QCheckBox
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from acquisition.image_loader import ImageLoader, ImageLoadError
from acquisition.metadata import ImageMetadata
from acquisition.scale_manager import ScaleManager
from preprocessing.image_processor import ImageProcessor
from app.viewers.image_viewer import ImageViewer
from app.dialogs.metadata_dialog import MetadataDialog
from app.dialogs.scale_dialog import ScaleDialog


class MainWindow(QMainWindow):
    """Janela principal do ARGOS — Sistema de Análise Metalográfica."""

    def __init__(self):
        super().__init__()

        # Estado da aplicação
        self._original_image: Optional[np.ndarray] = None
        self._metadata: Optional[ImageMetadata] = None
        self._scale_manager = ScaleManager()
        self._processor: Optional[ImageProcessor] = None

        self.setWindowTitle("ARGOS — Análise Metalográfica")
        self.setMinimumSize(1100, 700)

        self._build_ui()
        self._apply_styles()
        self._update_controls_state()

    # ──────────────────────────────────────────────
    # Construção da UI
    # ──────────────────────────────────────────────

    def _build_ui(self):
        """Monta todos os componentes da interface."""
        self._build_toolbar()
        self._build_central_widget()
        self._build_status_bar()

    def _build_toolbar(self):
        """Barra de ferramentas superior."""
        toolbar = QToolBar("Principal")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        self.act_open = QAction("📂  Abrir Imagem", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.setToolTip("Abrir imagem metalográfica (Ctrl+O)")
        self.act_open.triggered.connect(self._on_open_image)
        toolbar.addAction(self.act_open)

        toolbar.addSeparator()

        self.act_scale = QAction("⚖️  Calibrar Escala", self)
        self.act_scale.setToolTip("Definir escala µm/pixel")
        self.act_scale.triggered.connect(self._on_calibrate_scale)
        toolbar.addAction(self.act_scale)

        self.act_metadata = QAction("📋  Metadados", self)
        self.act_metadata.setToolTip("Editar metadados da amostra")
        self.act_metadata.triggered.connect(self._on_edit_metadata)
        toolbar.addAction(self.act_metadata)

        toolbar.addSeparator()

        self.act_reset = QAction("↺  Resetar Filtros", self)
        self.act_reset.setToolTip("Desliga todos os filtros, volta à imagem original")
        self.act_reset.triggered.connect(self._on_reset)
        toolbar.addAction(self.act_reset)

    def _build_central_widget(self):
        """Widget central com painel de controles + visualizadores."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([320, 800])
        splitter.setHandleWidth(2)

        main_layout.addWidget(splitter)

    # ──────────────────────────────────────────────
    # Widget reutilizável: Checkbox + Slider
    # ──────────────────────────────────────────────

    def _build_toggle_slider(
        self,
        label_text: str,
        min_val: int,
        max_val: int,
        default_val: int,
        on_toggled: Callable[[bool], None],
        on_value_changed: Callable[[int], None],
        value_formatter: Callable[[int], str] = str,
    ) -> tuple[QCheckBox, QSlider, QWidget]:
        """
        Cria um controle padrão de filtro: checkbox (liga/desliga) +
        slider (intensidade), com label de valor ao lado do checkbox.

        Comportamento:
            • Slider começa desabilitado (filtro desligado por padrão)
            • Marcar o checkbox habilita o slider e dispara on_toggled(True)
            • Desmarcar desabilita o slider e dispara on_toggled(False)
            • Mover o slider só dispara on_value_changed() se o
              checkbox estiver marcado (evita recálculo com filtro inativo)

        Args:
            label_text: Nome do filtro, exibido no checkbox.
            min_val, max_val, default_val: Range do slider (inteiros;
                para valores fracionários, escale no callback, ex: v/100).
            on_toggled: Chamado com True/False quando o checkbox muda.
            on_value_changed: Chamado com o valor do slider quando ele
                muda E o checkbox está marcado.
            value_formatter: Formata o valor numérico para exibição
                (ex: lambda v: f"{v/100:.2f}x" para contraste).

        Returns:
            (checkbox, slider, container_widget) — o container já vem
            pronto para addWidget() no layout do grupo.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        checkbox = QCheckBox(label_text)
        value_label = QLabel(value_formatter(default_val))
        value_label.setStyleSheet("color: #6c9bcf; font-size: 11px;")
        header.addWidget(checkbox)
        header.addStretch()
        header.addWidget(value_label)
        layout.addLayout(header)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setEnabled(False)
        layout.addWidget(slider)

        def _handle_toggle(checked: bool):
            slider.setEnabled(checked)
            on_toggled(checked)

        def _handle_value(v: int):
            value_label.setText(value_formatter(v))
            if checkbox.isChecked():
                on_value_changed(v)

        checkbox.toggled.connect(_handle_toggle)
        slider.valueChanged.connect(_handle_value)

        return checkbox, slider, container

    # ──────────────────────────────────────────────
    # Painel esquerdo — grupos de filtro
    # ──────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        """Painel esquerdo com controles de pré-processamento."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("PRÉ-PROCESSAMENTO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 11px; font-weight: bold; letter-spacing: 2px; "
            "color: #4a6fa5; padding: 4px;"
        )
        layout.addWidget(title)

        layout.addWidget(self._build_grayscale_group())
        layout.addWidget(self._build_brightness_contrast_group())
        layout.addWidget(self._build_clahe_group())
        layout.addWidget(self._build_smoothing_group())
        layout.addWidget(self._build_edge_group())

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_grayscale_group(self) -> QGroupBox:
        """Grayscale não tem parâmetro — só checkbox liga/desliga."""
        group = QGroupBox("Conversão")
        layout = QVBoxLayout(group)

        self.chk_grayscale = QCheckBox("Escala de Cinza")
        self.chk_grayscale.toggled.connect(self._on_grayscale_toggled)
        layout.addWidget(self.chk_grayscale)

        return group

    def _build_brightness_contrast_group(self) -> QGroupBox:
        group = QGroupBox("Brilho e Contraste")
        layout = QVBoxLayout(group)

        self.chk_brightness, self.sld_brightness, w1 = self._build_toggle_slider(
            "Brilho", -127, 127, 0,
            on_toggled=self._on_brightness_toggled,
            on_value_changed=self._on_brightness_value,
        )
        layout.addWidget(w1)

        self.chk_contrast, self.sld_contrast, w2 = self._build_toggle_slider(
            "Contraste", 10, 300, 100,
            on_toggled=self._on_contrast_toggled,
            on_value_changed=self._on_contrast_value,
            value_formatter=lambda v: f"{v/100:.2f}×",
        )
        layout.addWidget(w2)

        return group

    def _build_clahe_group(self) -> QGroupBox:
        group = QGroupBox("CLAHE — Contraste Local")
        layout = QVBoxLayout(group)

        self.chk_clahe, self.sld_clahe, w = self._build_toggle_slider(
            "CLAHE", 10, 100, 20,
            on_toggled=self._on_clahe_toggled,
            on_value_changed=self._on_clahe_value,
            value_formatter=lambda v: f"clip={v/10:.1f}",
        )
        layout.addWidget(w)

        return group

    def _build_smoothing_group(self) -> QGroupBox:
        """
        Os três filtros de suavização são independentes entre si —
        o usuário pode ligar mais de um ao mesmo tempo (são aplicados
        em sequência fixa: Gaussiano → Mediana → Bilateral).
        """
        group = QGroupBox("Filtros de Suavização")
        layout = QVBoxLayout(group)

        self.chk_gaussian, self.sld_gaussian, w1 = self._build_toggle_slider(
            "Gaussiano", 1, 15, 5,
            on_toggled=self._on_gaussian_toggled,
            on_value_changed=self._on_gaussian_value,
            value_formatter=lambda v: f"k={v if v % 2 else v + 1}",
        )
        layout.addWidget(w1)

        self.chk_median, self.sld_median, w2 = self._build_toggle_slider(
            "Mediana", 1, 15, 5,
            on_toggled=self._on_median_toggled,
            on_value_changed=self._on_median_value,
            value_formatter=lambda v: f"k={v if v % 2 else v + 1}",
        )
        layout.addWidget(w2)

        self.chk_bilateral, self.sld_bilateral, w3 = self._build_toggle_slider(
            "Bilateral", 1, 25, 9,
            on_toggled=self._on_bilateral_toggled,
            on_value_changed=self._on_bilateral_value,
            value_formatter=lambda v: f"d={v}",
        )
        layout.addWidget(w3)

        return group

    def _build_edge_group(self) -> QGroupBox:
        group = QGroupBox("Realce de Bordas")
        layout = QVBoxLayout(group)

        self.chk_canny, self.sld_canny, w1 = self._build_toggle_slider(
            "Canny", 10, 200, 50,
            on_toggled=self._on_canny_toggled,
            on_value_changed=self._on_canny_value,
            value_formatter=lambda v: f"t1={v}",
        )
        layout.addWidget(w1)

        self.chk_sobel, self.sld_sobel, w2 = self._build_toggle_slider(
            "Sobel", 1, 3, 1,
            on_toggled=self._on_sobel_toggled,
            on_value_changed=self._on_sobel_value,
            value_formatter=lambda v: f"ksize={2*v+1}",
        )
        layout.addWidget(w2)

        return group

    # ──────────────────────────────────────────────
    # Painel direito e status bar (iguais à versão anterior)
    # ──────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)

        orig_widget = QWidget()
        orig_layout = QVBoxLayout(orig_widget)
        orig_layout.setContentsMargins(0, 0, 0, 0)
        orig_lbl = QLabel("ORIGINAL")
        orig_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_lbl.setStyleSheet("color: #4a6fa5; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        self.viewer_original = ImageViewer("Abra uma imagem para começar")
        orig_layout.addWidget(orig_lbl)
        orig_layout.addWidget(self.viewer_original)
        splitter.addWidget(orig_widget)

        proc_widget = QWidget()
        proc_layout = QVBoxLayout(proc_widget)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_lbl = QLabel("PROCESSADA")
        proc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        proc_lbl.setStyleSheet("color: #06d6a0; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        self.viewer_processed = ImageViewer("Ative um filtro para ver o resultado")
        proc_layout.addWidget(proc_lbl)
        proc_layout.addWidget(self.viewer_processed)
        splitter.addWidget(proc_widget)

        layout.addWidget(splitter)
        return widget

    def _build_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)

        self.status_file = QLabel("Nenhuma imagem carregada")
        self.status_scale = QLabel("Escala: não calibrada")
        self.status_scale.setStyleSheet("color: #f4a261;")
        self.status_dims = QLabel("")
        self.status_ops = QLabel("")
        self.status_ops.setStyleSheet("color: #06d6a0;")

        status.addWidget(self.status_file)
        status.addPermanentWidget(self._separator())
        status.addPermanentWidget(self.status_scale)
        status.addPermanentWidget(self._separator())
        status.addPermanentWidget(self.status_dims)
        status.addPermanentWidget(self._separator())
        status.addPermanentWidget(self.status_ops)

    def _separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #2a2a4a;")
        return sep

    # ──────────────────────────────────────────────
    # Ações da toolbar
    # ──────────────────────────────────────────────

    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir Imagem Metalográfica",
            "",
            "Imagens (*.tiff *.tif *.png *.jpg *.jpeg);;Todos os arquivos (*)"
        )
        if not path:
            return

        try:
            loader = ImageLoader()
            image, metadata = loader.load(path)

            self._original_image = image
            self._metadata = metadata
            self._processor = ImageProcessor(image)
            self._scale_manager = ScaleManager()

            self._reset_all_controls()

            self.viewer_original.set_image(image)
            self.viewer_processed.set_image(image)

            self.status_file.setText(f"📄 {metadata.file_name}")
            self.status_dims.setText(f"{metadata.width_px} × {metadata.height_px} px")
            self._update_scale_status()
            self._update_controls_state()

            dlg = MetadataDialog(self._metadata, self)
            dlg.exec()

        except ImageLoadError as e:
            QMessageBox.critical(self, "Erro ao carregar imagem", str(e))

    def _on_calibrate_scale(self):
        if self._metadata is None:
            QMessageBox.warning(self, "ARGOS", "Carregue uma imagem primeiro.")
            return

        dlg = ScaleDialog(self._scale_manager, self)
        if dlg.exec():
            self._metadata.scale_um_per_px = self._scale_manager.um_per_px
            self._update_scale_status()

    def _on_edit_metadata(self):
        if self._metadata is None:
            QMessageBox.warning(self, "ARGOS", "Carregue uma imagem primeiro.")
            return
        dlg = MetadataDialog(self._metadata, self)
        dlg.exec()

    def _on_reset(self):
        """Desliga todos os filtros (desmarca checkboxes) e reseta o processor."""
        if self._processor is None:
            return
        self._reset_all_controls()
        self._processor.reset()
        self.viewer_processed.set_image(self._processor.current_image)
        self.status_ops.setText("↺ Reset")

    def _reset_all_controls(self):
        """Desmarca todos os checkboxes sem disparar recálculo repetido."""
        for chk in (
            self.chk_grayscale, self.chk_brightness, self.chk_contrast,
            self.chk_clahe, self.chk_gaussian, self.chk_median,
            self.chk_bilateral, self.chk_canny, self.chk_sobel,
        ):
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        for sld in (
            self.sld_brightness, self.sld_contrast, self.sld_clahe,
            self.sld_gaussian, self.sld_median, self.sld_bilateral,
            self.sld_canny, self.sld_sobel,
        ):
            sld.setEnabled(False)

    # ──────────────────────────────────────────────
    # Callbacks de filtro — cada um atualiza o estado
    # do ImageProcessor e chama _on_state_changed()
    # ──────────────────────────────────────────────

    def _on_grayscale_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.grayscale_enabled = checked
        self._on_state_changed("Grayscale")

    def _on_brightness_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.brightness_enabled = checked
        self._on_state_changed("Brilho")

    def _on_brightness_value(self, v: int):
        if self._processor is None:
            return
        self._processor.brightness_value = v
        self._on_state_changed("Brilho")

    def _on_contrast_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.contrast_enabled = checked
        self._on_state_changed("Contraste")

    def _on_contrast_value(self, v: int):
        if self._processor is None:
            return
        self._processor.contrast_value = v / 100.0
        self._on_state_changed("Contraste")

    def _on_clahe_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.clahe_enabled = checked
        self._on_state_changed("CLAHE")

    def _on_clahe_value(self, v: int):
        if self._processor is None:
            return
        self._processor.clahe_clip = v / 10.0
        self._on_state_changed("CLAHE")

    def _on_gaussian_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.gaussian_enabled = checked
        self._on_state_changed("Gaussiano")

    def _on_gaussian_value(self, v: int):
        if self._processor is None:
            return
        self._processor.gaussian_kernel = v if v % 2 else v + 1
        self._on_state_changed("Gaussiano")

    def _on_median_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.median_enabled = checked
        self._on_state_changed("Mediana")

    def _on_median_value(self, v: int):
        if self._processor is None:
            return
        self._processor.median_kernel = v if v % 2 else v + 1
        self._on_state_changed("Mediana")

    def _on_bilateral_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.bilateral_enabled = checked
        self._on_state_changed("Bilateral")

    def _on_bilateral_value(self, v: int):
        if self._processor is None:
            return
        self._processor.bilateral_d = v
        self._on_state_changed("Bilateral")

    def _on_canny_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.canny_enabled = checked
        self._on_state_changed("Canny")

    def _on_canny_value(self, v: int):
        if self._processor is None:
            return
        # threshold2 mantido proporcional (3x), padrão comum para Canny
        self._processor.canny_threshold1 = float(v)
        self._processor.canny_threshold2 = float(v * 3)
        self._on_state_changed("Canny")

    def _on_sobel_toggled(self, checked: bool):
        if self._processor is None:
            return
        self._processor.sobel_enabled = checked
        self._on_state_changed("Sobel")

    def _on_sobel_value(self, v: int):
        if self._processor is None:
            return
        self._processor.sobel_ksize = 2 * v + 1  # 1→3, 2→5, 3→7
        self._on_state_changed("Sobel")

    def _on_state_changed(self, label: str):
        """
        Ponto único de recálculo: qualquer mudança de checkbox ou slider
        passa por aqui. Recalcula o pipeline inteiro e atualiza a tela.
        """
        if self._processor is None:
            return
        try:
            self._processor.rebuild()
            self.viewer_processed.set_image(self._processor.current_image)
            ops = len(self._processor.operation_log)
            self.status_ops.setText(f"✔ {label}  ({ops} filtro{'s' if ops != 1 else ''} ativo{'s' if ops != 1 else ''})")
        except Exception as e:
            QMessageBox.warning(self, "Erro ao aplicar filtro", str(e))

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _update_controls_state(self):
        has_image = self._original_image is not None
        self.act_scale.setEnabled(has_image)
        self.act_metadata.setEnabled(has_image)
        self.act_reset.setEnabled(has_image)

    def _update_scale_status(self):
        if self._scale_manager.is_calibrated:
            self.status_scale.setText(
                f"⚖️  {self._scale_manager.um_per_px:.4f} µm/px"
            )
            self.status_scale.setStyleSheet("color: #06d6a0;")
        else:
            self.status_scale.setText("⚠️  Escala: não calibrada")
            self.status_scale.setStyleSheet("color: #f4a261;")

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a1a; }

            QToolBar {
                background-color: #0f0f23;
                border-bottom: 1px solid #1a1a3a;
                padding: 4px;
                spacing: 4px;
            }
            QToolButton {
                background-color: transparent;
                color: #a0a0c0;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #1a1a3a;
                border-color: #2a2a5a;
                color: #e0e0ff;
            }
            QToolButton:pressed {
                background-color: #00526e;
            }

            QSplitter::handle { background-color: #1a1a3a; }

            QScrollArea { border: none; background-color: #0d0d20; }
            QWidget { background-color: #0a0a1a; }

            QGroupBox {
                color: #6c7fa5;
                border: 1px solid #1a1a3a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                color: #4a6fa5;
            }

            QLabel { color: #8090b0; font-size: 12px; }

            QCheckBox { color: #c0c0e0; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #2a2a4a;
                border-radius: 3px;
                background-color: #131330;
            }
            QCheckBox::indicator:checked {
                background-color: #00b4d8;
                border-color: #00b4d8;
            }

            QSlider::groove:horizontal {
                height: 4px;
                background: #1a1a3a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:disabled {
                background: #333355;
            }
            QSlider::sub-page:horizontal {
                background: #004d65;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal:disabled {
                background: #1a1a3a;
            }

            QPushButton {
                background-color: #131330;
                color: #8090c0;
                border: 1px solid #1e1e40;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #1a1a4a;
                color: #c0d0ff;
                border-color: #2a2a6a;
            }
            QPushButton:pressed {
                background-color: #00526e;
                color: white;
            }
            QPushButton:disabled {
                color: #333355;
                border-color: #141428;
            }

            QStatusBar {
                background-color: #070714;
                color: #5060a0;
                font-size: 11px;
                border-top: 1px solid #1a1a3a;
            }
            QStatusBar QLabel {
                color: #5060a0;
                font-size: 11px;
                padding: 0 8px;
            }
        """)