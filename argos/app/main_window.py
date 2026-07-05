"""
app/main_window.py

Janela principal do ARGOS.
Conecta todos os módulos: aquisição → pré-processamento → visualização.

Layout:
    ┌─────────────────────────────────────────────┐
    │  Toolbar (Abrir, Escala, Metadados)          │
    ├──────────────────────┬──────────────────────┤
    │   Painel Esquerdo    │   Painel Direito      │
    │   (Controles de      │   (Original | Atual)  │
    │    filtros)          │                       │
    ├──────────────────────┴──────────────────────┤
    │  Status Bar (escala, dimensões, operações)   │
    └─────────────────────────────────────────────┘
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QStatusBar, QLabel, QSlider,
    QPushButton, QGroupBox, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QFrame, QCheckBox
)
from PyQt6.QtGui import QAction, QIcon, QFont
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
        self._original_image: np.ndarray | None = None
        self._metadata: ImageMetadata | None = None
        self._scale_manager = ScaleManager()
        self._processor: ImageProcessor | None = None

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

        # Ação: Abrir imagem
        self.act_open = QAction("📂  Abrir Imagem", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.setToolTip("Abrir imagem metalográfica (Ctrl+O)")
        self.act_open.triggered.connect(self._on_open_image)
        toolbar.addAction(self.act_open)

        toolbar.addSeparator()

        # Ação: Calibrar escala
        self.act_scale = QAction("⚖️  Calibrar Escala", self)
        self.act_scale.setToolTip("Definir escala µm/pixel")
        self.act_scale.triggered.connect(self._on_calibrate_scale)
        toolbar.addAction(self.act_scale)

        # Ação: Metadados
        self.act_metadata = QAction("📋  Metadados", self)
        self.act_metadata.setToolTip("Editar metadados da amostra")
        self.act_metadata.triggered.connect(self._on_edit_metadata)
        toolbar.addAction(self.act_metadata)

        toolbar.addSeparator()

        # Ação: Reset processamento
        self.act_reset = QAction("↺  Resetar Filtros", self)
        self.act_reset.setToolTip("Desfaz todos os filtros, volta à imagem original")
        self.act_reset.triggered.connect(self._on_reset)
        toolbar.addAction(self.act_reset)

    def _build_central_widget(self):
        """Widget central com painel de controles + visualizadores."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter divide painel esquerdo (controles) e direito (imagens)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Painel esquerdo: controles de filtro
        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        # Painel direito: visualizadores
        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 800])
        splitter.setHandleWidth(2)

        main_layout.addWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        """Painel esquerdo com controles de pré-processamento."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- Título ---
        title = QLabel("PRÉ-PROCESSAMENTO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 11px; font-weight: bold; letter-spacing: 2px; "
            "color: #4a6fa5; padding: 4px;"
        )
        layout.addWidget(title)

        # --- Grupo: Conversão ---
        layout.addWidget(self._build_grayscale_group())

        # --- Grupo: Brilho / Contraste ---
        layout.addWidget(self._build_brightness_group())

        # --- Grupo: Equalização CLAHE ---
        layout.addWidget(self._build_clahe_group())

        # --- Grupo: Filtros de suavização ---
        layout.addWidget(self._build_blur_group())

        # --- Grupo: Realce de bordas ---
        layout.addWidget(self._build_edge_group())

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_grayscale_group(self) -> QGroupBox:
        group = QGroupBox("Conversão")
        layout = QVBoxLayout(group)

        self.btn_grayscale = QPushButton("Converter para Escala de Cinza")
        self.btn_grayscale.clicked.connect(self._on_apply_grayscale)
        layout.addWidget(self.btn_grayscale)

        return group

    def _build_brightness_group(self) -> QGroupBox:
        group = QGroupBox("Brilho e Contraste")
        layout = QVBoxLayout(group)

        # Brilho
        layout.addWidget(QLabel("Brilho:"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-127, 127)
        self.brightness_slider.setValue(0)
        self.brightness_label = QLabel("0")
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(str(v))
        )
        row = QHBoxLayout()
        row.addWidget(self.brightness_slider)
        row.addWidget(self.brightness_label)
        layout.addLayout(row)

        # Contraste
        layout.addWidget(QLabel("Contraste:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 300)  # 0.1x a 3.0x (×0.01)
        self.contrast_slider.setValue(100)
        self.contrast_label = QLabel("1.00×")
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(f"{v/100:.2f}×")
        )
        row2 = QHBoxLayout()
        row2.addWidget(self.contrast_slider)
        row2.addWidget(self.contrast_label)
        layout.addLayout(row2)

        btn = QPushButton("▶  Aplicar")
        btn.clicked.connect(self._on_apply_brightness_contrast)
        layout.addWidget(btn)

        return group

    def _build_clahe_group(self) -> QGroupBox:
        group = QGroupBox("CLAHE — Contraste Local")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Clip Limit:"))
        self.clahe_clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_clip_slider.setRange(10, 100)  # 1.0 a 10.0 (×0.1)
        self.clahe_clip_slider.setValue(20)
        self.clahe_clip_label = QLabel("2.0")
        self.clahe_clip_slider.valueChanged.connect(
            lambda v: self.clahe_clip_label.setText(f"{v/10:.1f}")
        )
        row = QHBoxLayout()
        row.addWidget(self.clahe_clip_slider)
        row.addWidget(self.clahe_clip_label)
        layout.addLayout(row)

        btn = QPushButton("▶  Aplicar CLAHE")
        btn.clicked.connect(self._on_apply_clahe)
        layout.addWidget(btn)

        return group

    def _build_blur_group(self) -> QGroupBox:
        group = QGroupBox("Filtros de Suavização")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Tamanho do kernel:"))
        self.blur_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_kernel_slider.setRange(1, 15)
        self.blur_kernel_slider.setValue(5)
        self.blur_kernel_label = QLabel("5")
        self.blur_kernel_slider.valueChanged.connect(
            lambda v: self.blur_kernel_label.setText(str(v if v % 2 == 1 else v + 1))
        )
        row = QHBoxLayout()
        row.addWidget(self.blur_kernel_slider)
        row.addWidget(self.blur_kernel_label)
        layout.addLayout(row)

        btn_gauss = QPushButton("▶  Gaussiano")
        btn_gauss.clicked.connect(self._on_apply_gaussian)
        btn_gauss.setToolTip("Suavização geral — reduz todo tipo de ruído")
        layout.addWidget(btn_gauss)

        btn_median = QPushButton("▶  Mediana")
        btn_median.clicked.connect(self._on_apply_median)
        btn_median.setToolTip("Remove ruído granular — preserva bordas")
        layout.addWidget(btn_median)

        btn_bilateral = QPushButton("▶  Bilateral")
        btn_bilateral.clicked.connect(self._on_apply_bilateral)
        btn_bilateral.setToolTip("Suaviza preservando contornos de grão")
        layout.addWidget(btn_bilateral)

        return group

    def _build_edge_group(self) -> QGroupBox:
        group = QGroupBox("Realce de Bordas")
        layout = QVBoxLayout(group)

        btn_canny = QPushButton("▶  Canny")
        btn_canny.clicked.connect(self._on_apply_canny)
        btn_canny.setToolTip("Detecção de bordas binárias")
        layout.addWidget(btn_canny)

        btn_sobel = QPushButton("▶  Sobel")
        btn_sobel.clicked.connect(self._on_apply_sobel)
        btn_sobel.setToolTip("Gradiente de bordas (intensidade proporcional)")
        layout.addWidget(btn_sobel)

        return group

    def _build_right_panel(self) -> QWidget:
        """Painel direito: imagem original (topo) e imagem processada (baixo)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Labels de título
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Original
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

        # Processada
        proc_widget = QWidget()
        proc_layout = QVBoxLayout(proc_widget)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_lbl = QLabel("PROCESSADA")
        proc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        proc_lbl.setStyleSheet("color: #06d6a0; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        self.viewer_processed = ImageViewer("Aplique um filtro para ver o resultado")
        proc_layout.addWidget(proc_lbl)
        proc_layout.addWidget(self.viewer_processed)
        splitter.addWidget(proc_widget)

        layout.addWidget(splitter)
        return widget

    def _build_status_bar(self):
        """Barra de status inferior com informações da imagem."""
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
    # Ações (slots conectados à toolbar / botões)
    # ──────────────────────────────────────────────

    def _on_open_image(self):
        """Abre diálogo de arquivo e carrega a imagem selecionada."""
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

            # Exibe a imagem original
            self.viewer_original.set_image(image)
            self.viewer_processed.set_image(image)

            # Atualiza status
            self.status_file.setText(f"📄 {metadata.file_name}")
            self.status_dims.setText(f"{metadata.width_px} × {metadata.height_px} px")
            self._update_scale_status()
            self._update_controls_state()

            # Abre diálogo de metadados automaticamente
            dlg = MetadataDialog(self._metadata, self)
            dlg.exec()

        except ImageLoadError as e:
            QMessageBox.critical(self, "Erro ao carregar imagem", str(e))

    def _on_calibrate_scale(self):
        """Abre diálogo de calibração de escala."""
        if self._metadata is None:
            QMessageBox.warning(self, "ARGOS", "Carregue uma imagem primeiro.")
            return

        dlg = ScaleDialog(self._scale_manager, self)
        if dlg.exec():
            self._metadata.scale_um_per_px = self._scale_manager.um_per_px
            self._update_scale_status()

    def _on_edit_metadata(self):
        """Abre diálogo de metadados."""
        if self._metadata is None:
            QMessageBox.warning(self, "ARGOS", "Carregue uma imagem primeiro.")
            return
        dlg = MetadataDialog(self._metadata, self)
        dlg.exec()

    def _on_reset(self):
        """Reseta todos os filtros, volta à imagem original."""
        if self._processor is None:
            return
        self._processor.reset()
        self.viewer_processed.set_image(self._processor.current_image)
        self.status_ops.setText("↺ Reset")

    # --- Filtros ---

    def _on_apply_grayscale(self):
        self._apply_filter(lambda: self._processor.apply_grayscale(), "Grayscale")

    def _on_apply_brightness_contrast(self):
        b = self.brightness_slider.value()
        c = self.contrast_slider.value() / 100.0
        self._apply_filter(
            lambda: self._processor.apply_brightness_contrast(b, c),
            f"Brilho={b}, Contraste={c:.2f}×"
        )

    def _on_apply_clahe(self):
        clip = self.clahe_clip_slider.value() / 10.0
        self._apply_filter(
            lambda: self._processor.apply_clahe(clip_limit=clip),
            f"CLAHE(clip={clip})"
        )

    def _on_apply_gaussian(self):
        k = self.blur_kernel_slider.value()
        if k % 2 == 0:
            k += 1
        self._apply_filter(
            lambda: self._processor.apply_gaussian(kernel_size=k),
            f"Gaussian(k={k})"
        )

    def _on_apply_median(self):
        k = self.blur_kernel_slider.value()
        if k % 2 == 0:
            k += 1
        self._apply_filter(
            lambda: self._processor.apply_median(kernel_size=k),
            f"Median(k={k})"
        )

    def _on_apply_bilateral(self):
        self._apply_filter(
            lambda: self._processor.apply_bilateral(),
            "Bilateral"
        )

    def _on_apply_canny(self):
        self._apply_filter(
            lambda: self._processor.apply_canny(),
            "Canny"
        )

    def _on_apply_sobel(self):
        self._apply_filter(
            lambda: self._processor.apply_sobel(),
            "Sobel"
        )

    def _apply_filter(self, fn, label: str):
        """
        Método genérico para aplicar qualquer filtro.
        Evita repetição de código nos métodos acima.

        Args:
            fn: Função lambda que chama o filtro no processor.
            label: Nome para mostrar na status bar.
        """
        if self._processor is None:
            QMessageBox.warning(self, "ARGOS", "Carregue uma imagem primeiro.")
            return
        try:
            fn()
            self.viewer_processed.set_image(self._processor.current_image)
            ops = len(self._processor.operation_log)
            self.status_ops.setText(f"✔ {label}  ({ops} op{'s' if ops > 1 else ''})")
        except Exception as e:
            QMessageBox.warning(self, "Erro ao aplicar filtro", str(e))

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _update_controls_state(self):
        """Habilita/desabilita controles conforme estado."""
        has_image = self._original_image is not None
        self.act_scale.setEnabled(has_image)
        self.act_metadata.setEnabled(has_image)
        self.act_reset.setEnabled(has_image)

    def _update_scale_status(self):
        """Atualiza o label de escala na status bar."""
        if self._scale_manager.is_calibrated:
            self.status_scale.setText(
                f"⚖️  {self._scale_manager.um_per_px:.4f} µm/px"
            )
            self.status_scale.setStyleSheet("color: #06d6a0;")
        else:
            self.status_scale.setText("⚠️  Escala: não calibrada")
            self.status_scale.setStyleSheet("color: #f4a261;")

    def _apply_styles(self):
        """Estilo global da janela principal."""
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
            QSlider::sub-page:horizontal {
                background: #004d65;
                border-radius: 2px;
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
