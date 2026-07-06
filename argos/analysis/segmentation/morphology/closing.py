"""
analysis/segmentation/morphology/closing.py

Fechamento morfológico (closing) = Dilatação seguida de Erosão.
É o "espelho" da abertura — ordem invertida das mesmas duas operações.

Por que essa ordem importa:
    A dilatação sozinha fecha buracos pretos dentro dos objetos, mas
    engorda o objeto inteiro. Ao erodir em seguida com o MESMO kernel,
    o objeto recupera aproximadamente o tamanho original — mas os buracos
    pretos internos que a dilatação já preencheu não reaparecem.

Efeito final:
    Fecha pequenos buracos pretos DENTRO dos objetos e une pequenas
    fissuras na borda, sem alterar significativamente o tamanho externo
    do objeto.

Quando usar em metalografia:
    Depois do threshold, quando um grão aparece com "furos" internos
    falsos — geralmente causados por reflexo de luz ou variação sutil
    de ataque químico dentro da própria fase. Fechamento resolve isso
    sem precisar mexer manualmente em cada grão.

    Sequência típica recomendada: Opening (limpa ruído externo) →
    Closing (fecha buracos internos) → measurement (agora sim mede).

Equivalente no OpenCV:
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    Aqui implementamos explicitamente como dilate→erode para reaproveitar
    as classes Dilation e Erosion já existentes (mesmo kernel, rastreável).
"""

import numpy as np

from analysis.segmentation.morphology.erosion import Erosion
from analysis.segmentation.morphology.dilation import Dilation


class Closing:
    """
    Aplica fechamento morfológico (dilatação → erosão) em uma máscara binária.

    Uso:
        op = Closing(kernel_size=3, iterations=1)
        mascara_sem_furos = op.apply(mascara_binaria)
    """

    def __init__(self, kernel_size: int = 3, iterations: int = 1, kernel_shape: str = "ellipse"):
        """
        Args:
            kernel_size: Tamanho do elemento estruturante (ímpar). Padrão: 3.
            iterations: Repetições do par dilatação/erosão. Padrão: 1.
            kernel_shape: "rect", "ellipse" ou "cross". Padrão: "ellipse".
        """
        self._dilation = Dilation(kernel_size, iterations, kernel_shape)
        self._erosion = Erosion(kernel_size, iterations, kernel_shape)

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Aplica fechamento: dilata primeiro, depois erode com o mesmo kernel.

        Args:
            mask: Máscara binária (0/255), 2D.

        Returns:
            Máscara com buracos internos preenchidos.
        """
        dilated = self._dilation.apply(mask)
        return self._erosion.apply(dilated)

    def __str__(self) -> str:
        return (
            f"Closing(kernel={self._dilation.kernel_size}x{self._dilation.kernel_size}, "
            f"shape={self._dilation.kernel_shape}, iterations={self._dilation.iterations})"
        )