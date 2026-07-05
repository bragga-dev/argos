"""
acquisition/metadata.py

Responsável por armazenar e registrar todos os metadados de uma imagem metalográfica.
Metadados são informações sobre a imagem — não a imagem em si.
Exemplos: aumento do microscópio, material, ataque químico usado, operador, data.

Princípio ARGOS: nenhuma análise começa sem metadados registrados.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ImageMetadata:
    """
    Representa todos os metadados de uma amostra metalográfica.

    @dataclass é um decorador do Python que cria automaticamente
    os métodos __init__, __repr__, etc., para a classe.
    """

    # --- Informações da imagem ---
    file_path: str = ""                     # Caminho completo do arquivo
    file_name: str = ""                     # Nome do arquivo
    width_px: int = 0                       # Largura em pixels
    height_px: int = 0                      # Altura em pixels

    # --- Escala (OBRIGATÓRIO para medições reais) ---
    scale_um_per_px: Optional[float] = None  # Micrômetros por pixel (µm/px)
    scale_bar_px: Optional[float] = None     # Comprimento da barra de escala em pixels
    scale_bar_um: Optional[float] = None     # Comprimento da barra de escala em µm

    # --- Condições de aquisição ---
    magnification: Optional[int] = None     # Aumento (ex: 100x, 200x, 500x)
    objective_lens: Optional[str] = None    # Lente objetiva usada
    etchant: Optional[str] = None           # Ataque químico (ex: Nital 2%)
    etching_time_s: Optional[float] = None  # Tempo de ataque em segundos

    # --- Informações do material ---
    material: Optional[str] = None          # Material (ex: AISI 1045)
    heat_treatment: Optional[str] = None    # Tratamento térmico aplicado
    standard: Optional[str] = None         # Norma a aplicar (ex: ASTM E112)

    # --- Rastreabilidade ---
    operator: Optional[str] = None          # Nome do operador
    date_acquired: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None             # Observações livres

    def has_scale(self) -> bool:
        """Retorna True se a escala µm/pixel está definida."""
        return self.scale_um_per_px is not None and self.scale_um_per_px > 0

    def to_dict(self) -> dict:
        """Converte os metadados para dicionário (útil para salvar em log)."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "dimensions": f"{self.width_px} x {self.height_px} px",
            "scale_um_per_px": self.scale_um_per_px,
            "magnification": self.magnification,
            "etchant": self.etchant,
            "material": self.material,
            "standard": self.standard,
            "operator": self.operator,
            "date": self.date_acquired.isoformat(),
            "notes": self.notes,
        }

    def __str__(self) -> str:
        escala = (
            f"{self.scale_um_per_px:.4f} µm/px"
            if self.has_scale()
            else "NÃO DEFINIDA"
        )
        return (
            f"Arquivo: {self.file_name}\n"
            f"Dimensões: {self.width_px} x {self.height_px} px\n"
            f"Escala: {escala}\n"
            f"Aumento: {self.magnification}x\n"
            f"Material: {self.material}\n"
            f"Ataque: {self.etchant}\n"
            f"Operador: {self.operator}"
        )
