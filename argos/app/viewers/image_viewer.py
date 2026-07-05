"""
app/viewers/image_viewer.py

Widget PyQt6 para exibir imagens OpenCV/NumPy na interface gráfica.

O PyQt6 usa QPixmap para exibir imagens, mas o OpenCV usa arrays NumPy.
Este widget faz a conversão e exibe a imagem com zoom e pan.
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class ImageViewer(QLabel):
    """
    Widget para exibir imagens NumPy (OpenCV) no PyQt6.

    Herda de QLabel pois ele já sabe exibir QPixmap.
    Fazemos a conversão NumPy → QPixmap internamente.

    Uso:
        viewer = ImageViewer()
        viewer.set_image(minha_imagem_numpy)
    """

    def __init__(self, placeholder_text: str = "Nenhuma imagem carregada"):
        super().__init__()

        self.setText(placeholder_text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                color: #4a4a6a;
                border: 2px dashed #2a2a4a;
                border-radius: 8px;
                font-size: 14px;
            }
        """)

        self._original_pixmap: QPixmap | None = None

    def set_image(self, image: np.ndarray | None):
        """
        Exibe uma imagem NumPy no widget.

        Args:
            image: Array NumPy (grayscale 2D ou BGR 3D). None limpa o widget.
        """
        if image is None:
            self._original_pixmap = None
            self.setText("Nenhuma imagem carregada")
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a1a2e;
                    color: #4a4a6a;
                    border: 2px dashed #2a2a4a;
                    border-radius: 8px;
                    font-size: 14px;
                }
            """)
            return

        pixmap = self._numpy_to_pixmap(image)
        self._original_pixmap = pixmap
        self.setStyleSheet("""
            QLabel {
                background-color: #0d0d1a;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
            }
        """)
        self._update_display()

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """
        Converte array NumPy para QPixmap do PyQt6.

        Fluxo de conversão:
            NumPy array → QImage → QPixmap

        Por que precisamos disso?
            PyQt6 não sabe o que é NumPy. Precisamos converter para
            o formato que o Qt entende (QImage), depois para QPixmap.
        """
        # Caso 1: Imagem grayscale (2D)
        if len(image.shape) == 2:
            h, w = image.shape
            img_contiguous = np.ascontiguousarray(image)
            q_image = QImage(
                img_contiguous.data,
                w, h,
                w,  # bytes por linha = largura (1 byte por pixel em grayscale)
                QImage.Format.Format_Grayscale8
            )

        # Caso 2: Imagem colorida BGR (OpenCV usa BGR, não RGB!)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Converte BGR (OpenCV) → RGB (Qt)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            img_contiguous = np.ascontiguousarray(image_rgb)
            q_image = QImage(
                img_contiguous.data,
                w, h,
                w * ch,  # bytes por linha = largura × canais
                QImage.Format.Format_RGB888
            )

        # Caso 3: Imagem com canal alpha (BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            h, w, ch = image_rgba.shape
            img_contiguous = np.ascontiguousarray(image_rgba)
            q_image = QImage(
                img_contiguous.data,
                w, h,
                w * ch,
                QImage.Format.Format_RGBA8888
            )
        else:
            raise ValueError(f"Formato de imagem não suportado: shape={image.shape}")

        return QPixmap.fromImage(q_image)

    def _update_display(self):
        """Atualiza a exibição mantendo proporção e ajustando ao tamanho do widget."""
        if self._original_pixmap is None:
            return
        scaled = self._original_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Reescala a imagem quando o widget muda de tamanho."""
        super().resizeEvent(event)
        self._update_display()
