"""
preprocessing/filters/median.py

Filtro Mediana — remoção de ruído granular (sal e pimenta).

O que faz:
    Substitui cada pixel pelo VALOR MEDIANO dos seus vizinhos.
    Diferente do gaussiano, não "mistura" pixels — ele escolhe um valor real.

Por que é melhor que o gaussiano para metalografia:
    ✔ Remove ruído pontual (sal e pimenta) muito bem
    ✔ Preserva bordas e contornos de grão
    ✔ Ideal para imagens ruidosas de microscópio óptico

Quando usar:
    • Imagens com pontos brilhantes/escuros isolados
    • Antes da segmentação para não confundir ruído com contorno de grão
    • Como primeira escolha quando o ruído é claramente granular
"""

import cv2
import numpy as np


class MedianFilter:
    """
    Aplica filtro de mediana para remoção de ruído granular.

    Uso:
        f = MedianFilter(kernel_size=5)
        resultado = f.apply(imagem)
    """

    def __init__(self, kernel_size: int = 5):
        """
        Args:
            kernel_size: Tamanho do kernel (deve ser ímpar). Padrão: 5.
                3 → suave, preserva muito detalhe
                5 → equilíbrio (recomendado para começar)
                7 → mais agressivo
                9+ → remove muito detalhe, usar com cuidado
        """
        self.kernel_size = self._validate_kernel(kernel_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica o filtro de mediana.

        Args:
            image: Imagem de entrada (grayscale ou colorida).

        Returns:
            Imagem filtrada. A imagem original NÃO é modificada.

        Nota técnica: cv2.medianBlur exige que a imagem seja uint8
        para kernels > 5. Para segurança, fazemos a conversão internamente.
        """
        # Garante uint8 para compatibilidade com kernels maiores
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return cv2.medianBlur(image, self.kernel_size)

    def set_kernel_size(self, size: int):
        """Atualiza o tamanho do kernel com validação."""
        self.kernel_size = self._validate_kernel(size)

    @staticmethod
    def _validate_kernel(size: int) -> int:
        """Garante que o kernel seja ímpar e positivo."""
        if size < 1:
            size = 1
        if size % 2 == 0:
            size += 1
        return size

    def __str__(self) -> str:
        return f"MedianFilter(kernel={self.kernel_size}x{self.kernel_size})"
