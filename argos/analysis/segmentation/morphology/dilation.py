"""
analysis/segmentation/morphology/dilation.py

Dilatação morfológica — "expande" as regiões brancas (objeto) da máscara.
É o oposto da erosão.

O que faz:
    Desliza um elemento estruturante (kernel) pela imagem. Um pixel vira
    branco (255) se PELO MENOS UM pixel sob o kernel for branco.

Efeito prático:
    Preenche pequenos buracos pretos dentro dos objetos.
    Une objetos que estão muito próximos (às vezes indesejado — ver aviso).
    Engorda as bordas dos objetos.

Quando usar em metalografia:
    Depois de uma erosão, para recuperar o tamanho original do objeto sem
    trazer de volta o ruído que a erosão removeu — essa combinação
    erosão→dilatação é exatamente a operação de "abertura" (opening.py).

    Usado sozinho: para fechar pequenos buracos pretos dentro de um grão
    (ruído do ataque químico que criou um "furo" falso no meio do grão).

Cuidado:
    Dilatação excessiva funde grãos que deveriam ser contados separadamente,
    o que destrói a contagem de partículas e a distribuição granulométrica.
"""

import cv2
import numpy as np


class DilationError(Exception):
    """Erro de operação morfológica de dilatação."""
    pass


class Dilation:
    """
    Aplica dilatação morfológica em uma máscara binária.

    Uso:
        op = Dilation(kernel_size=3, iterations=1)
        mascara_dilatada = op.apply(mascara_binaria)
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
                positivo). Padrão: 3. Maior = dilatação mais agressiva.
            iterations: Quantas vezes a dilatação é aplicada em sequência.
                Padrão: 1.
            kernel_shape: "rect", "ellipse" ou "cross". "ellipse" costuma
                dar resultados mais naturais para grãos/partículas.
        """
        self.kernel_size = self._validate_kernel_size(kernel_size)
        self.iterations = max(1, int(iterations))
        self.kernel_shape = self._validate_shape(kernel_shape)

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Aplica dilatação na máscara.

        Args:
            mask: Máscara binária (0/255), 2D.

        Returns:
            Máscara dilatada, mesmo shape da entrada.
        """
        mask = self._ensure_binary(mask)
        kernel = cv2.getStructuringElement(
            self.KERNEL_SHAPES[self.kernel_shape],
            (self.kernel_size, self.kernel_size),
        )
        return cv2.dilate(mask, kernel, iterations=self.iterations)

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
            raise DilationError(
                f"kernel_shape deve ser 'rect', 'ellipse' ou 'cross'. Recebido: '{shape}'"
            )
        return shape

    def __str__(self) -> str:
        return (
            f"Dilation(kernel={self.kernel_size}x{self.kernel_size}, "
            f"shape={self.kernel_shape}, iterations={self.iterations})"
        )