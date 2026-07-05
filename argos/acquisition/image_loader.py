"""
acquisition/image_loader.py

Responsável por carregar imagens metalográficas do disco para a memória.
Esta é a PRIMEIRA camada do ARGOS — Aquisição.

Regras desta camada:
  ✔ Importar TIFF, PNG, JPEG
  ✔ Validar formato e integridade
  ✔ Registrar metadados básicos (dimensões, nome, caminho)
  ✗ Nenhum filtro ou processamento aqui
  ✗ Nenhuma modificação na imagem original
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from acquisition.metadata import ImageMetadata


# Formatos suportados pelo ARGOS
SUPPORTED_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg"}


class ImageLoadError(Exception):
    """Erro lançado quando a imagem não pode ser carregada."""
    pass


class ImageLoader:
    """
    Carrega imagens metalográficas com validação e registro de metadados.

    Uso básico:
        loader = ImageLoader()
        image, metadata = loader.load("caminho/para/imagem.tif")
    """

    def load(self, file_path: str) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Carrega uma imagem e retorna o array NumPy + metadados básicos.

        Args:
            file_path: Caminho completo para o arquivo de imagem.

        Returns:
            Tupla (imagem_numpy, metadados).

        Raises:
            ImageLoadError: Se o arquivo não existir, formato inválido, ou
                            imagem corrompida.

        O que é np.ndarray?
            É a estrutura de dados principal do NumPy.
            Uma imagem colorida é representada como um array 3D:
            (altura, largura, canais) → ex: (1024, 1024, 3) para BGR.
        """
        path = Path(file_path)

        # 1. Verifica se o arquivo existe
        if not path.exists():
            raise ImageLoadError(f"Arquivo não encontrado: {file_path}")

        # 2. Verifica se a extensão é suportada
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                f"Formato '{ext}' não suportado. "
                f"Use: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # 3. Carrega a imagem com OpenCV
        # cv2.IMREAD_UNCHANGED preserva: escala de cinza, 16-bit, alpha channel
        # Isso é importante para TIFF metalográficos de alta profundidade
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ImageLoadError(
                f"Não foi possível ler o arquivo: {file_path}\n"
                f"O arquivo pode estar corrompido ou em formato não suportado."
            )

        # 4. Normaliza para 8-bit se necessário (TIFF pode ser 16-bit)
        image = self._normalize_to_8bit(image)

        # 5. Monta os metadados básicos
        metadata = self._build_metadata(image, path)

        return image, metadata

    def _normalize_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """
        Converte imagem para 8-bit (0-255) se ela for 16-bit.

        Microscópios de alta qualidade geram imagens de 16-bit (0-65535).
        Para processamento com OpenCV precisamos de 8-bit.
        A normalização preserva o contraste relativo.
        """
        if image.dtype == np.uint16:
            # Normaliza o range completo de 16-bit para 8-bit
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        return image

    def _build_metadata(self, image: np.ndarray, path: Path) -> ImageMetadata:
        """
        Extrai metadados básicos da imagem carregada.
        A escala µm/pixel deve ser definida posteriormente pelo usuário.
        """
        # Dimensões: image.shape retorna (altura, largura) ou (altura, largura, canais)
        height = image.shape[0]
        width = image.shape[1]

        metadata = ImageMetadata(
            file_path=str(path.resolve()),
            file_name=path.name,
            width_px=width,
            height_px=height,
        )
        return metadata

    @staticmethod
    def is_grayscale(image: np.ndarray) -> bool:
        """Retorna True se a imagem é em escala de cinza (2D) ou False se colorida (3D)."""
        return len(image.shape) == 2

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Converte imagem colorida para escala de cinza.
        Em metalografia, trabalhamos principalmente em escala de cinza.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image  # já é escala de cinza

    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Retorna o tamanho do arquivo em MB."""
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
