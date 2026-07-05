"""
preprocessing/filters/clahe.py

CLAHE — Contrast Limited Adaptive Histogram Equalization
(Equalização de Histograma Adaptativa com Limite de Contraste)

O que faz:
    Melhora o contraste LOCAL da imagem.
    Divide a imagem em blocos e equaliza cada bloco separadamente.
    O "Contrast Limited" evita amplificar ruído demais.

Por que é essencial em metalografia:
    Ataques químicos produzem contrastes sutis e irregulares.
    A iluminação do microscópio raramente é perfeitamente uniforme.
    CLAHE revela detalhes microestruturais que seriam invisíveis no original.

Comparação:
    equalizeHist → equalização GLOBAL (às vezes estraga a imagem)
    CLAHE → equalização LOCAL (muito mais inteligente)

Parâmetros:
    clip_limit: quanto de amplificação máxima é permitida
        2.0 → suave (padrão, bom ponto de partida)
        4.0 → mais agressivo
    tile_grid_size: tamanho dos blocos locais
        (8, 8) → padrão, funciona bem para a maioria dos casos
"""

import cv2
import numpy as np


class CLAHEFilter:
    """
    Aplica equalização de histograma adaptativa com limite de contraste.

    Uso:
        f = CLAHEFilter(clip_limit=2.0)
        resultado = f.apply(imagem_grayscale)
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """
        Args:
            clip_limit: Limite de amplificação de contraste. Padrão: 2.0.
            tile_grid_size: Tamanho dos blocos para equalização local.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE na imagem.

        IMPORTANTE: CLAHE funciona apenas em imagens grayscale (1 canal).
        Se a imagem for colorida, convertemos para grayscale automaticamente.

        Args:
            image: Imagem de entrada.

        Returns:
            Imagem com contraste melhorado.
        """
        # Converte para grayscale se necessário
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return self._clahe.apply(image)

    def set_clip_limit(self, clip_limit: float):
        """Atualiza o limite de contraste e recria o objeto CLAHE."""
        self.clip_limit = clip_limit
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def set_tile_grid_size(self, size: tuple):
        """Atualiza o tamanho dos blocos e recria o objeto CLAHE."""
        self.tile_grid_size = size
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=size
        )

    def __str__(self) -> str:
        return (
            f"CLAHE("
            f"clip_limit={self.clip_limit}, "
            f"tile_grid={self.tile_grid_size})"
        )
