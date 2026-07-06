"""
analysis/segmentation/morphology/erosion.py

Erosão morfológica — "encolhe" as regiões brancas (objeto) da máscara.

O que faz:
    Desliza um elemento estruturante (kernel) pela imagem. Um pixel só
    permanece branco (255) se TODOS os pixels sob o kernel também forem
    brancos. Caso contrário, vira preto.

Efeito prático:
    Remove ruído pontual pequeno (pontos brancos isolados somem).
    Afina bordas dos objetos.
    Separa objetos que estão quase colados por uma ponte fina de pixels.

Quando usar em metalografia:
    Depois de um threshold que sobrou "sujeira" — pequenos pontos brancos
    espalhados que não são grão/fase de verdade, só ruído da segmentação.

Cuidado:
    Erosão excessiva pode apagar grãos pequenos legítimos, não só ruído.
    Sempre visualizar antes/depois antes de aceitar o resultado
    (princípio ARGOS: nada é automático sem visualização).
"""

import cv2
import numpy as np


class ErosionError(Exception):
    """Erro de operação morfológica de erosão."""
    pass


class Erosion:
    """
    Aplica erosão morfológica em uma máscara binária.

    Uso:
        op = Erosion(kernel_size=3, iterations=1)
        mascara_erodida = op.apply(mascara_binaria)
    """

    KERNEL_SHAPES = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }

    def __init__(self, kernel_size: int = 3, iterations: int = 1, kernel_shape: str = "ellipse"):
        """
        Args:
            kernel_size: Tamanho do elemento estruturante (deve ser ímpar,
                positivo). Padrão: 3. Maior = erosão mais agressiva.
            iterations: Quantas vezes a erosão é aplicada em sequência.
                Padrão: 1. Equivalente (mas não idêntico) a um kernel maior.
            kernel_shape: "rect", "ellipse" ou "cross". "ellipse" costuma
                dar resultados mais naturais para grãos/partículas.
        """
        self.kernel_size = self._validate_kernel_size(kernel_size)
        self.iterations = max(1, int(iterations))
        self.kernel_shape = self._validate_shape(kernel_shape)

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Aplica erosão na máscara.

        Args:
            mask: Máscara binária (0/255), 2D.

        Returns:
            Máscara erodida, mesmo shape da entrada.
        """
        mask = self._ensure_binary(mask)
        kernel = cv2.getStructuringElement(
            self.KERNEL_SHAPES[self.kernel_shape],
            (self.kernel_size, self.kernel_size),
        )
        return cv2.erode(mask, kernel, iterations=self.iterations)

    @staticmethod
    def _ensure_binary(mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        return mask

    @staticmethod
    def _validate_kernel_size(size: int) -> int:
        if size < 1:
            size = 1
        if size % 2 == 0:
            size += 1
        return size

    @classmethod
    def _validate_shape(cls, shape: str) -> str:
        shape = shape.lower()
        if shape not in cls.KERNEL_SHAPES:
            raise ErosionError(
                f"kernel_shape deve ser 'rect', 'ellipse' ou 'cross'. Recebido: '{shape}'"
            )
        return shape

    def __str__(self) -> str:
        return (
            f"Erosion(kernel={self.kernel_size}x{self.kernel_size}, "
            f"shape={self.kernel_shape}, iterations={self.iterations})"
        )