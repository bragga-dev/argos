"""
acquisition/scale_manager.py

Gerencia a calibração de escala: converte pixels em µm (micrômetros) reais.

Por que isso é crítico?
    Sem escala, todos os resultados são em pixels — inúteis para engenharia.
    Com escala, podemos medir: tamanho de grão, espaçamento de fases, inclusões.

Como calibrar:
    Método 1 — Barra de escala na imagem:
        O usuário mede a barra de escala em pixels na imagem.
        Ex: barra de 100µm ocupa 320px → escala = 100/320 = 0.3125 µm/px

    Método 2 — Aumento do microscópio:
        Usando fator de calibração do equipamento.
        Ex: 200x → cada pixel = X µm (depende da câmera e ocular)
"""

from typing import Optional


class ScaleCalibrationError(Exception):
    """Erro de calibração de escala."""
    pass


class ScaleManager:
    """
    Gerencia a escala µm/pixel da imagem.

    Uso:
        sm = ScaleManager()
        sm.calibrate_from_bar(bar_length_px=320, bar_length_um=100)
        print(sm.um_per_px)   # → 0.3125
        print(sm.px_to_um(500))  # → 156.25 µm
    """

    def __init__(self):
        self._um_per_px: Optional[float] = None  # Escala principal
        self._calibration_method: Optional[str] = None

    @property
    def um_per_px(self) -> Optional[float]:
        """Retorna a escala em µm por pixel."""
        return self._um_per_px

    @property
    def is_calibrated(self) -> bool:
        """True se a escala está definida."""
        return self._um_per_px is not None and self._um_per_px > 0

    def calibrate_from_bar(self, bar_length_px: float, bar_length_um: float) -> float:
        """
        Calibra pela barra de escala presente na imagem.

        Args:
            bar_length_px: Comprimento da barra em pixels (medido na imagem).
            bar_length_um: Comprimento real da barra em µm (impresso na barra).

        Returns:
            Escala calculada em µm/pixel.

        Exemplo:
            calibrate_from_bar(320, 100) → 0.3125 µm/px
        """
        if bar_length_px <= 0:
            raise ScaleCalibrationError("Comprimento da barra em pixels deve ser > 0.")
        if bar_length_um <= 0:
            raise ScaleCalibrationError("Comprimento real da barra deve ser > 0.")

        self._um_per_px = bar_length_um / bar_length_px
        self._calibration_method = f"Barra de escala: {bar_length_um}µm = {bar_length_px}px"
        return self._um_per_px

    def calibrate_from_value(self, um_per_px: float) -> float:
        """
        Define a escala diretamente (quando o usuário já conhece o valor).

        Args:
            um_per_px: Escala em micrômetros por pixel.
        """
        if um_per_px <= 0:
            raise ScaleCalibrationError("A escala µm/px deve ser um valor positivo.")

        self._um_per_px = um_per_px
        self._calibration_method = f"Valor direto: {um_per_px} µm/px"
        return self._um_per_px

    def px_to_um(self, pixels: float) -> float:
        """
        Converte pixels para micrômetros.

        Args:
            pixels: Valor em pixels.

        Returns:
            Valor em µm.
        """
        self._require_calibration()
        return pixels * self._um_per_px

    def um_to_px(self, micrometers: float) -> float:
        """
        Converte micrômetros para pixels.

        Args:
            micrometers: Valor em µm.

        Returns:
            Valor em pixels.
        """
        self._require_calibration()
        return micrometers / self._um_per_px

    def px2_to_um2(self, area_px: float) -> float:
        """
        Converte área em pixels² para µm².
        Para área, o fator é quadrático: µm² = px² × (µm/px)²
        """
        self._require_calibration()
        return area_px * (self._um_per_px ** 2)

    def _require_calibration(self):
        """Lança erro se a escala ainda não foi definida."""
        if not self.is_calibrated:
            raise ScaleCalibrationError(
                "Escala não calibrada. "
                "Use calibrate_from_bar() ou calibrate_from_value() primeiro."
            )

    def __str__(self) -> str:
        if self.is_calibrated:
            return (
                f"Escala: {self._um_per_px:.4f} µm/px | "
                f"Método: {self._calibration_method}"
            )
        return "Escala: NÃO CALIBRADA"
