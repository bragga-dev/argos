"""
preprocessing/image_processor.py

Orquestra todos os filtros de pré-processamento.
É o ponto central da Camada 2 — Preparação.

Responsabilidades:
  ✔ Aplicar filtros na ordem correta
  ✔ Manter referência à imagem original (nunca destruir)
  ✔ Registrar parâmetros de cada operação (rastreabilidade)
  ✔ Permitir reset para estado anterior
"""

from typing import Optional
import numpy as np
import cv2

from preprocessing.filters.gaussian_blur import GaussianBlurFilter
from preprocessing.filters.median import MedianFilter
from preprocessing.filters.bilateral import BilateralFilter
from preprocessing.filters.clahe import CLAHEFilter


class ImageProcessor:
    """
    Gerencia o pré-processamento de imagens metalográficas.

    Armazena a imagem original intocada e a imagem atual (processada).
    Todos os filtros são aplicados sobre a imagem atual.

    Uso:
        processor = ImageProcessor(imagem_original)
        processor.apply_grayscale()
        processor.apply_clahe(clip_limit=2.0)
        processor.apply_median(kernel_size=5)
        resultado = processor.current_image
    """

    def __init__(self, original: np.ndarray):
        """
        Args:
            original: Imagem original carregada pelo ImageLoader.
                      Esta imagem NUNCA será modificada.
        """
        self._original: np.ndarray = original.copy()
        self._current: np.ndarray = original.copy()
        self._log: list[str] = []  # Log de operações aplicadas

    @property
    def original_image(self) -> np.ndarray:
        """Retorna a imagem original (somente leitura)."""
        return self._original.copy()

    @property
    def current_image(self) -> np.ndarray:
        """Retorna a imagem com os processamentos aplicados até agora."""
        return self._current.copy()

    @property
    def operation_log(self) -> list[str]:
        """Retorna o histórico de operações aplicadas."""
        return self._log.copy()

    def reset(self):
        """Desfaz TODOS os processamentos — volta à imagem original."""
        self._current = self._original.copy()
        self._log.append(">>> RESET: voltou para imagem original")

    # ──────────────────────────────────────────────
    # Conversão
    # ──────────────────────────────────────────────

    def apply_grayscale(self):
        """Converte a imagem para escala de cinza."""
        if len(self._current.shape) == 3:
            self._current = cv2.cvtColor(self._current, cv2.COLOR_BGR2GRAY)
            self._log.append("Convertida para grayscale")

    def apply_brightness_contrast(self, brightness: int = 0, contrast: float = 1.0):
        """
        Ajusta brilho e contraste da imagem.

        Fórmula: output = contrast × input + brightness

        Args:
            brightness: Deslocamento de brilho. Range: -127 a +127.
                        0 = sem mudança.
            contrast: Multiplicador de contraste. Range: 0.0 a 3.0.
                      1.0 = sem mudança. >1 = mais contraste.
        """
        self._current = cv2.convertScaleAbs(
            self._current, alpha=contrast, beta=brightness
        )
        self._log.append(
            f"Brilho/Contraste: brightness={brightness}, contrast={contrast:.2f}"
        )

    # ──────────────────────────────────────────────
    # Filtros
    # ──────────────────────────────────────────────

    def apply_gaussian(self, kernel_size: int = 5, sigma: float = 0):
        """Aplica filtro gaussiano."""
        f = GaussianBlurFilter(kernel_size=kernel_size, sigma=sigma)
        self._current = f.apply(self._current)
        self._log.append(str(f))

    def apply_median(self, kernel_size: int = 5):
        """Aplica filtro de mediana."""
        f = MedianFilter(kernel_size=kernel_size)
        self._current = f.apply(self._current)
        self._log.append(str(f))

    def apply_bilateral(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        """Aplica filtro bilateral."""
        f = BilateralFilter(d=d, sigma_color=sigma_color, sigma_space=sigma_space)
        self._current = f.apply(self._current)
        self._log.append(str(f))

    def apply_clahe(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """Aplica CLAHE para melhoria de contraste local."""
        f = CLAHEFilter(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        self._current = f.apply(self._current)
        self._log.append(str(f))

    # ──────────────────────────────────────────────
    # Realce de bordas
    # ──────────────────────────────────────────────

    def apply_canny(self, threshold1: float = 50, threshold2: float = 150):
        """
        Detecção de bordas com algoritmo de Canny.
        Resultado: imagem binária com bordas em branco.

        threshold1/threshold2: limiares para histerese.
        """
        gray = self._current
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        self._current = cv2.Canny(gray, threshold1, threshold2)
        self._log.append(f"Canny(t1={threshold1}, t2={threshold2})")

    def apply_sobel(self, ksize: int = 3):
        """
        Calcula gradiente de Sobel (magnitude das bordas).
        Mostra bordas com intensidade proporcional ao contraste.
        """
        gray = self._current
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normaliza para 0-255
        self._current = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self._log.append(f"Sobel(ksize={ksize})")
