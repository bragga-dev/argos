"""
preprocessing/filters/gaussian_blur.py

Filtro Gaussiano — suavização geral da imagem.

O que faz:
    Reduz ruído substituindo cada pixel pela média ponderada dos seus vizinhos.
    Pixels mais próximos têm mais peso (distribuição gaussiana = forma de sino).

Quando usar em metalografia:
    • Antes do threshold para evitar falsos contornos por ruído
    • Para suavizar gradientes suaves de iluminação
    • NÃO usar quando as bordas finas dos grãos são importantes

Parâmetro principal:
    kernel_size: tamanho da "janela" de suavização.
    Deve ser ímpar: 3, 5, 7, 9, 11...
    Maior → mais suave, mas perde mais detalhe.
"""

import cv2
import numpy as np


class GaussianBlurFilter:
    """
    Aplica suavização gaussiana de forma controlada e reversível.

    Uso:
        f = GaussianBlurFilter(kernel_size=5)
        resultado = f.apply(imagem)
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 0):
        """
        Args:
            kernel_size: Tamanho do kernel (deve ser ímpar). Padrão: 5.
            sigma: Desvio padrão gaussiano. 0 = calculado automaticamente.
        """
        self.kernel_size = self._validate_kernel(kernel_size)
        self.sigma = sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica o filtro gaussiano na imagem.

        Args:
            image: Imagem de entrada (grayscale ou colorida).

        Returns:
            Imagem suavizada (mesmo tipo e shape que a entrada).

        Nota: A imagem original NÃO é modificada — retornamos uma cópia.
        """
        return cv2.GaussianBlur(
            image,
            ksize=(self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )

    def set_kernel_size(self, size: int):
        """Atualiza o tamanho do kernel com validação."""
        self.kernel_size = self._validate_kernel(size)

    @staticmethod
    def _validate_kernel(size: int) -> int:
        """Garante que o kernel seja ímpar e positivo."""
        if size < 1:
            size = 1
        if size % 2 == 0:
            size += 1  # Torna ímpar automaticamente
        return size

    def __str__(self) -> str:
        return f"GaussianBlur(kernel={self.kernel_size}x{self.kernel_size}, sigma={self.sigma})"
