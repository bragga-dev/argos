from pathlib import Path
from PIL import Image, UnidentifiedImageError
import logging

logger = logging.getLogger(__name__)

VALID_FORMATS = {"JPEG", "PNG", "WEBP"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_SIZE_MB = 5
MAX_WIDTH = 4000
MAX_HEIGHT = 4000


class ImageValidationError(Exception):
    """Erro de validação de imagem."""
    pass


def validate_image_file(file):
    """
    Valida uma imagem.

    Aceita:
        - pathlib.Path
        - string contendo caminho
        - arquivo aberto em modo binário
        - io.BytesIO

    Retorna True se for válida.
    """

    # --------------------------
    # Descobre nome e tamanho
    # --------------------------

    if isinstance(file, (str, Path)):
        path = Path(file)

        if not path.exists():
            raise FileNotFoundError(path)

        name = path.name
        size = path.stat().st_size

        f = open(path, "rb")
        close_after = True

    else:
        f = file
        close_after = False

        name = getattr(file, "name", "")
        current_pos = file.tell()

        file.seek(0, 2)
        size = file.tell()
        file.seek(current_pos)

    try:

        # --------------------------
        # Extensão
        # --------------------------

        ext = Path(name).suffix.lower()

        if ext not in VALID_EXTENSIONS:
            raise ImageValidationError(
                f"Extensão '{ext}' inválida. "
                f"Permitidas: {', '.join(sorted(VALID_EXTENSIONS))}"
            )

        # --------------------------
        # Tamanho
        # --------------------------

        if size > MAX_SIZE_MB * 1024 * 1024:
            raise ImageValidationError(
                f"Arquivo possui {size / (1024 * 1024):.1f} MB "
                f"(máximo {MAX_SIZE_MB} MB)"
            )

        # --------------------------
        # Integridade
        # --------------------------

        f.seek(0)

        try:
            img = Image.open(f)
            img.verify()

        except UnidentifiedImageError:
            raise ImageValidationError("Arquivo não é uma imagem válida.")

        f.seek(0)

        # --------------------------
        # Reabre
        # --------------------------

        img = Image.open(f)

        fmt = (img.format or "").upper()

        if fmt not in VALID_FORMATS:
            raise ImageValidationError(
                f"Formato '{fmt}' não suportado."
            )

        width, height = img.size

        if width > MAX_WIDTH or height > MAX_HEIGHT:
            raise ImageValidationError(
                f"Imagem possui {width}x{height}px "
                f"(máximo {MAX_WIDTH}x{MAX_HEIGHT})"
            )

        if width * height > MAX_WIDTH * MAX_HEIGHT:
            raise ImageValidationError(
                "Imagem possui pixels demais."
            )

        return True

    finally:

        if close_after:
            f.close()
        else:
            f.seek(0)