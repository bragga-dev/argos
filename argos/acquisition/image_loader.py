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
from validators.validate_image_file import validate_image_file, ImageValidationError

import cv2
import numpy as np

from acquisition.metadata import ImageMetadata

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

        A validação é feita ANTES do carregamento para garantir que:
            - O arquivo não está corrompido
            - O formato é suportado
            - A imagem não excede limites de tamanho e dimensões

        Args:
            file_path: Caminho completo para o arquivo de imagem.

        Returns:
            Tupla (imagem_numpy, metadados).

        Raises:
            ImageLoadError: Se o arquivo não existir, formato inválido, imagem corrompida
            ImageValidationError: Se a imagem não passar nos critérios de validação
        """
        
        path = Path(file_path)

        # 1. Verifica se o arquivo existe
        if not path.exists():
            raise ImageLoadError(f"Arquivo não encontrado: {file_path}")

        # 2. VALIDAÇÃO - Usa a função existente
        #    A função já verifica: extensão, tamanho, integridade, formato e dimensões
        try:
            is_valid = validate_image_file(file_path)
            if not is_valid:
                # Teoricamente não chega aqui, pois a função levanta exceção
                raise ImageValidationError("Validação falhou por motivo desconhecido.")
        except ImageValidationError as e:
            # Relança com contexto mais claro
            raise ImageLoadError(f"Falha na validação da imagem: {e}")
        except FileNotFoundError as e:
            # Já foi verificado, mas pode ocorrer em race condition
            raise ImageLoadError(f"Arquivo não encontrado: {e}")
        except Exception as e:
            # Captura qualquer outro erro inesperado da validação
            raise ImageLoadError(f"Erro inesperado na validação: {e}")

        # 3. Carrega a imagem com OpenCV (já sabemos que é válida)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        # 4. Normaliza para 8-bit se necessário
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
