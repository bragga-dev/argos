"""
preprocessing/filters/bilateral.py

Filtro Bilateral — suavização que preserva bordas.

O que faz:
    Combina dois critérios ao suavizar:
    1. Proximidade espacial (como o gaussiano)
    2. Similaridade de intensidade (novo!)

    Pixels distantes em INTENSIDADE não influenciam a suavização,
    mesmo que estejam espacialmente próximos.

Por que é valioso em metalografia:
    ✔ Suaviza o interior dos grãos
    ✔ Preserva os contornos de grão (alto contraste)
    ✔ Melhor que o gaussiano quando bordas são importantes

Desvantagem:
    É mais lento que gaussiano e mediana.
    Para imagens grandes, pode demorar alguns segundos.

Parâmetros:
    d: diâmetro da vizinhança de cada pixel
    sigma_color: tolerância de intensidade (quanto de diferença aceita)
    sigma_space: alcance espacial (como kernel gaussiano)
"""

import cv2
import numpy as np


class BilateralFilter:
    """
    Aplica filtro bilateral: suaviza preservando bordas.

    Uso:
        f = BilateralFilter(d=9, sigma_color=75, sigma_space=75)
        resultado = f.apply(imagem)
    """

    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        """
        Args:
            d: Diâmetro da vizinhança. -1 = calculado a partir de sigma_space.
               Valores típicos: 5, 7, 9. Maior → mais lento.
            sigma_color: Quanto de variação de intensidade é aceita.
               Pequeno (ex: 25): apenas pixels quase iguais influenciam.
               Grande (ex: 150): pixels bem diferentes também influenciam.
            sigma_space: Alcance espacial. Similar ao sigma do gaussiano.
               Maior = área maior de influência.
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica o filtro bilateral.

        Args:
            image: Imagem de entrada (grayscale ou colorida).

        Returns:
            Imagem filtrada com bordas preservadas.
        """
        # O bilateral do OpenCV exige uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return cv2.bilateralFilter(
            image,
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
        )

    def __str__(self) -> str:
        return (
            f"BilateralFilter("
            f"d={self.d}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space})"
        )
