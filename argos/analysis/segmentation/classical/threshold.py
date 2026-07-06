"""
analysis/segmentation/classical/threshold.py

Segmentação por limiarização (thresholding) — primeira e mais básica técnica
de segmentação. Divide a imagem em duas regiões: fundo e objeto (grão/fase),
com base em um valor de intensidade de corte.

Duas variantes:
    ThresholdManual — o usuário define o valor de corte (0-255).
        Útil quando o contraste é bem definido e conhecido.

    ThresholdOtsu — o valor de corte é calculado automaticamente pelo
        algoritmo de Otsu, que maximiza a separação entre as duas classes
        de intensidade (fundo vs objeto) com base no histograma.

Por que threshold é a base de tudo em análise metalográfica:
    Toda medição (área, contagem, tamanho de grão) depende de uma máscara
    binária que separa "o que conta" de "o que não conta". Threshold é o
    método mais simples e mais rápido de gerar essa máscara.

Regras desta camada (Análise — não confundir com Preparação):
    ✔ Recebe imagem já pré-processada (grayscale, filtrada)
    ✔ Retorna máscara binária (0 = fundo, 255 = objeto)
    ✔ Nunca decide por conta própria: threshold_value é sempre exposto
       e registrado, para rastreabilidade
    ✗ Não modifica a imagem original — trabalha sobre uma cópia
    ✗ Não mede nada — isso é responsabilidade de analysis/measurement/
"""

import cv2
import numpy as np


class ThresholdError(Exception):
    """Erro de segmentação por threshold."""
    pass


class ThresholdManual:
    """
    Segmentação por limiarização manual (valor fixo definido pelo usuário).

    Uso:
        t = ThresholdManual(threshold_value=128, invert=False)
        mascara = t.apply(imagem_grayscale)
    """

    def __init__(self, threshold_value: int = 128, invert: bool = False):
        """
        Args:
            threshold_value: Valor de corte (0-255). Pixels acima ficam
                brancos (255), pixels abaixo ficam pretos (0).
            invert: Se True, inverte o resultado (THRESH_BINARY_INV).
                Útil quando o grão de interesse é mais escuro que o fundo.
        """
        self.threshold_value = self._validate_threshold(threshold_value)
        self.invert = invert

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica threshold binário fixo.

        Args:
            image: Imagem grayscale (2D). Se vier colorida, é convertida
                automaticamente.

        Returns:
            Máscara binária (0 ou 255), mesmo shape 2D da entrada.
        """
        gray = self._ensure_grayscale(image)

        mode = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, self.threshold_value, 255, mode)
        return mask

    def set_threshold(self, value: int):
        """Atualiza o valor de corte com validação."""
        self.threshold_value = self._validate_threshold(value)

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    @staticmethod
    def _validate_threshold(value: int) -> int:
        if value < 0 or value > 255:
            raise ThresholdError(
                f"threshold_value deve estar entre 0 e 255. Recebido: {value}"
            )
        return int(value)

    def __str__(self) -> str:
        return f"ThresholdManual(value={self.threshold_value}, invert={self.invert})"


class ThresholdOtsu:
    """
    Segmentação por limiarização automática (método de Otsu).

    O algoritmo escolhe o valor de corte que minimiza a variância dentro
    de cada classe (fundo/objeto) e maximiza a variância entre classes.
    Funciona bem quando o histograma da imagem é bimodal (dois picos claros).

    Uso:
        t = ThresholdOtsu(invert=False)
        mascara = t.apply(imagem_grayscale)
        print(t.computed_threshold)  # valor que o Otsu calculou
    """

    def __init__(self, invert: bool = False):
        """
        Args:
            invert: Se True, inverte o resultado (THRESH_BINARY_INV).
        """
        self.invert = invert
        self._computed_threshold: float | None = None

    @property
    def computed_threshold(self) -> float | None:
        """
        Valor de corte calculado pelo Otsu na última chamada de apply().
        None se apply() ainda não foi chamado.

        Importante para rastreabilidade: mesmo sendo automático, o valor
        deve ser registrado no log de operações.
        """
        return self._computed_threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica threshold de Otsu.

        Args:
            image: Imagem grayscale (2D). Se vier colorida, é convertida
                automaticamente.

        Returns:
            Máscara binária (0 ou 255), mesmo shape 2D da entrada.
        """
        gray = ThresholdManual._ensure_grayscale(image)

        mode = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        mode |= cv2.THRESH_OTSU

        computed_value, mask = cv2.threshold(gray, 0, 255, mode)
        self._computed_threshold = computed_value
        return mask

    def __str__(self) -> str:
        val = f"{self._computed_threshold:.1f}" if self._computed_threshold else "—"
        return f"ThresholdOtsu(computed={val}, invert={self.invert})"