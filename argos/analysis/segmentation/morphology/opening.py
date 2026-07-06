"""
analysis/segmentation/morphology/opening.py

Abertura morfológica (opening) = Erosão seguida de Dilatação.

Por que essa ordem importa:
    A erosão sozinha remove ruído mas encolhe os objetos de verdade também.
    Ao dilatar em seguida com o MESMO kernel, os objetos que sobreviveram
    à erosão recuperam aproximadamente o tamanho original — mas o ruído
    pontual que a erosão já apagou não volta (porque não existe mais nada
    lá para dilatar).

Efeito final:
    Remove ruído pontual pequeno (pontos brancos isolados) SEM alterar
    significativamente o tamanho dos objetos reais.

Quando usar em metalografia:
    É a operação mais usada logo após o threshold, antes de qualquer
    medição. Limpa "sal" (pontos brancos espúrios) que o threshold sempre
    gera em imagens com textura ou ruído de captura.

Equivalente no OpenCV:
    cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    Aqui implementamos explicitamente como erode→dilate para reaproveitar
    as classes Erosion e Dilation já existentes (mesmo kernel, rastreável).
"""

import numpy as np

from analysis.segmentation.morphology.erosion import Erosion
from analysis.segmentation.morphology.dilation import Dilation


class Opening:
    """
    Aplica abertura morfológica (erosão → dilatação) em uma máscara binária.

    Uso:
        op = Opening(kernel_size=3, iterations=1)
        mascara_limpa = op.apply(mascara_binaria)
    """

    def __init__(self, kernel_size: int = 3, iterations: int = 1, kernel_shape: str = "ellipse"):
        """
        Args:
            kernel_size: Tamanho do elemento estruturante (ímpar). Padrão: 3.
            iterations: Repetições do par erosão/dilatação. Padrão: 1.
            kernel_shape: "rect", "ellipse" ou "cross". Padrão: "ellipse".
        """
        self._erosion = Erosion(kernel_size, iterations, kernel_shape)
        self._dilation = Dilation(kernel_size, iterations, kernel_shape)

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Aplica abertura: erode primeiro, depois dilata com o mesmo kernel.

        Args:
            mask: Máscara binária (0/255), 2D.

        Returns:
            Máscara com ruído pontual removido.
        """
        eroded = self._erosion.apply(mask)
        return self._dilation.apply(eroded)

    def __str__(self) -> str:
        return (
            f"Opening(kernel={self._erosion.kernel_size}x{self._erosion.kernel_size}, "
            f"shape={self._erosion.kernel_shape}, iterations={self._erosion.iterations})"
        )