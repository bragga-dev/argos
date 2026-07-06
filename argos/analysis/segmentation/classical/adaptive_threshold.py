"""
analysis/segmentation/classical/adaptive_threshold.py

Segmentação por limiarização adaptativa.

Problema que resolve:
    O ThresholdManual e o ThresholdOtsu usam UM valor de corte para a
    imagem inteira. Isso falha quando a iluminação não é uniforme —
    muito comum em microscopia óptica, onde o centro do campo de visão
    costuma ser mais claro que as bordas.

    Resultado sem adaptive threshold: metade da imagem fica bem segmentada,
    a outra metade vira tudo preto ou tudo branco.

Como resolve:
    Em vez de um valor global, calcula um valor de corte LOCAL para cada
    pixel, com base na vizinhança dele (uma janela de block_size x block_size).

    Duas formas de calcular a média local:
        MEAN    → média aritmética simples da vizinhança
        GAUSSIAN → média ponderada pela distância (pixels mais centrais pesam mais)

Parâmetros críticos:
    block_size: tamanho da vizinhança usada para calcular o limiar local.
        Deve ser ímpar. Pequeno (ex: 11) → sensível a detalhes finos.
        Grande (ex: 51) → mais robusto a ruído, mas mais lento e mais
        parecido com um threshold global.

    C: constante subtraída do valor médio local. Ajusta fino o quão
        "generoso" é o corte. Positivo → menos pixels classificados como
        objeto. Negativo → mais pixels classificados como objeto.

Regras desta camada (iguais ao threshold.py):
    ✔ Recebe imagem grayscale já pré-processada
    ✔ Retorna máscara binária (0 = fundo, 255 = objeto)
    ✔ Parâmetros sempre expostos e registrados (rastreabilidade)
    ✗ Não modifica a imagem de entrada
"""

import cv2
import numpy as np


class AdaptiveThresholdError(Exception):
    """Erro de segmentação por threshold adaptativo."""
    pass


class ThresholdAdaptive:
    """
    Segmentação por limiarização adaptativa (limiar local, não global).

    Uso:
        t = ThresholdAdaptive(block_size=25, c=5, method="gaussian")
        mascara = t.apply(imagem_grayscale)
    """

    METHODS = {
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    }

    def __init__(
        self,
        block_size: int = 25,
        c: float = 5,
        method: str = "gaussian",
        invert: bool = False,
    ):
        """
        Args:
            block_size: Tamanho da vizinhança local (deve ser ímpar, > 1).
                Padrão: 25.
            c: Constante subtraída da média local. Positivo = mais restritivo.
                Padrão: 5.
            method: "mean" (média simples) ou "gaussian" (média ponderada).
                Padrão: "gaussian" — geralmente dá bordas mais suaves.
            invert: Se True, inverte o resultado. Útil quando o grão de
                interesse é mais escuro que o fundo.
        """
        self.block_size = self._validate_block_size(block_size)
        self.c = c
        self.method = self._validate_method(method)
        self.invert = invert

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica threshold adaptativo.

        Args:
            image: Imagem grayscale (2D). Se vier colorida, é convertida
                automaticamente.

        Returns:
            Máscara binária (0 ou 255), mesmo shape 2D da entrada.
        """
        gray = self._ensure_grayscale(image)

        threshold_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        adaptive_method = self.METHODS[self.method]

        mask = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=adaptive_method,
            thresholdType=threshold_type,
            blockSize=self.block_size,
            C=self.c,
        )
        return mask

    def set_block_size(self, size: int):
        """Atualiza o tamanho do bloco com validação (deve ser ímpar > 1)."""
        self.block_size = self._validate_block_size(size)

    def set_c(self, c: float):
        """Atualiza a constante C."""
        self.c = c

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    @staticmethod
    def _validate_block_size(size: int) -> int:
        if size < 3:
            size = 3
        if size % 2 == 0:
            size += 1  # block_size precisa ser ímpar
        return size

    @classmethod
    def _validate_method(cls, method: str) -> str:
        method = method.lower()
        if method not in cls.METHODS:
            raise AdaptiveThresholdError(
                f"method deve ser 'mean' ou 'gaussian'. Recebido: '{method}'"
            )
        return method

    def __str__(self) -> str:
        return (
            f"ThresholdAdaptive(block_size={self.block_size}, "
            f"c={self.c}, method={self.method}, invert={self.invert})"
        )