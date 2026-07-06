"""
preprocessing/image_processor.py

Orquestra todos os filtros de pré-processamento.
É o ponto central da Camada 2 — Preparação.

MUDANÇA DE ARQUITETURA (v2):
    A versão anterior empilhava operações: cada apply_x() rodava em cima
    do resultado da anterior, e não havia como "desligar" um filtro já
    aplicado sem resetar tudo.

    Esta versão é um PIPELINE DECLARATIVO: o processor guarda o ESTADO de
    cada filtro (ligado/desligado + parâmetro atual), e o método rebuild()
    recalcula a imagem processada DO ZERO a partir da original, aplicando
    apenas os filtros que estão enabled=True, sempre na mesma ordem fixa.

    Por que isso importa:
        Com controles do tipo checkbox + slider, o usuário liga/desliga e
        ajusta valores livremente. Se cada mudança empilhasse sobre a
        imagem já processada, desligar um filtro não teria como reverter
        o efeito — o dado já estaria destruído. Recalcular sempre a partir
        do original garante reversibilidade real (princípio central do
        ARGOS: "nada é automático sem visualização, tudo é reversível").

    Custo: cada mudança reprocessa a imagem inteira. Para imagens de
    microscopia (tipicamente < 4000x4000), isso ainda roda em milissegundos
    com OpenCV. Se no futuro isso pesar, dá pra cachear estágios
    intermediários — não é necessário agora.

Ordem fixa do pipeline (não é reordenável pelo usuário nesta fase):
    1. Grayscale
    2. Brilho
    3. Contraste
    4. CLAHE
    5. Gaussiano
    6. Mediana
    7. Bilateral
    8. Canny
    9. Sobel
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
    Pipeline declarativo de pré-processamento.

    Cada filtro tem um estado `<nome>_enabled: bool` e, quando aplicável,
    um ou mais parâmetros (`<nome>_<param>`). A UI só precisa:
        1. Alterar o atributo de estado correspondente
        2. Chamar rebuild()
        3. Ler current_image

    Uso:
        processor = ImageProcessor(imagem_original)
        processor.grayscale_enabled = True
        processor.clahe_enabled = True
        processor.clahe_clip = 3.0
        processor.rebuild()
        resultado = processor.current_image
    """

    def __init__(self, original: np.ndarray):
        """
        Args:
            original: Imagem original carregada pelo ImageLoader.
                      Esta imagem NUNCA será modificada.
        """
        self._original: np.ndarray = original.copy()

        # --- Conversão ---
        self.grayscale_enabled: bool = False

        # --- Brilho / Contraste (dois filtros independentes) ---
        self.brightness_enabled: bool = False
        self.brightness_value: int = 0        # -127 a 127

        self.contrast_enabled: bool = False
        self.contrast_value: float = 1.0      # 0.1 a 3.0

        # --- CLAHE ---
        self.clahe_enabled: bool = False
        self.clahe_clip: float = 2.0
        self.clahe_tile_grid: tuple = (8, 8)

        # --- Suavização (independentes entre si — podem ser combinados) ---
        self.gaussian_enabled: bool = False
        self.gaussian_kernel: int = 5

        self.median_enabled: bool = False
        self.median_kernel: int = 5

        self.bilateral_enabled: bool = False
        self.bilateral_d: int = 9
        self.bilateral_sigma_color: float = 75
        self.bilateral_sigma_space: float = 75

        # --- Realce de bordas ---
        self.canny_enabled: bool = False
        self.canny_threshold1: float = 50
        self.canny_threshold2: float = 150

        self.sobel_enabled: bool = False
        self.sobel_ksize: int = 3

        self._current: np.ndarray = self._original.copy()
        self._log: list[str] = []
        self.rebuild()

    @property
    def original_image(self) -> np.ndarray:
        """Retorna a imagem original (somente leitura)."""
        return self._original.copy()

    @property
    def current_image(self) -> np.ndarray:
        """Retorna a imagem com o pipeline atual aplicado."""
        return self._current.copy()

    @property
    def operation_log(self) -> list[str]:
        """Lista dos filtros atualmente ativos, na ordem de aplicação."""
        return self._log.copy()

    def reset(self):
        """Desliga todos os filtros e volta aos parâmetros padrão."""
        original = self._original
        self.__init__(original)
        self._log = [">>> RESET: todos os filtros desligados"]

    def rebuild(self):
        """
        Recalcula a imagem processada do zero, a partir da original,
        aplicando apenas os filtros com enabled=True, na ordem fixa.

        Deve ser chamado toda vez que qualquer estado (enabled ou
        parâmetro) é alterado pela UI.
        """
        img = self._original.copy()
        log: list[str] = []

        if self.grayscale_enabled:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            log.append("Grayscale")

        if self.brightness_enabled:
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=self.brightness_value)
            log.append(f"Brilho({self.brightness_value})")

        if self.contrast_enabled:
            img = cv2.convertScaleAbs(img, alpha=self.contrast_value, beta=0)
            log.append(f"Contraste({self.contrast_value:.2f}x)")

        if self.clahe_enabled:
            f = CLAHEFilter(clip_limit=self.clahe_clip, tile_grid_size=self.clahe_tile_grid)
            img = f.apply(img)
            log.append(str(f))

        if self.gaussian_enabled:
            f = GaussianBlurFilter(kernel_size=self.gaussian_kernel)
            img = f.apply(img)
            log.append(str(f))

        if self.median_enabled:
            f = MedianFilter(kernel_size=self.median_kernel)
            img = f.apply(img)
            log.append(str(f))

        if self.bilateral_enabled:
            f = BilateralFilter(
                d=self.bilateral_d,
                sigma_color=self.bilateral_sigma_color,
                sigma_space=self.bilateral_sigma_space,
            )
            img = f.apply(img)
            log.append(str(f))

        if self.canny_enabled:
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
            log.append(f"Canny(t1={self.canny_threshold1:.0f}, t2={self.canny_threshold2:.0f})")

        if self.sobel_enabled:
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            log.append(f"Sobel(ksize={self.sobel_ksize})")

        self._current = img
        self._log = log if log else ["(nenhum filtro ativo — imagem original)"]