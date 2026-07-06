from analysis.segmentation.classical.threshold import ThresholdManual, ThresholdOtsu
from analysis.segmentation.classical.adaptive_threshold import ThresholdAdaptive
from analysis.segmentation.morphology.opening import Opening
from analysis.segmentation.morphology.closing import Closing
from analysis.segmentation.morphology.erosion import Erosion
from analysis.segmentation.morphology.dilation import Dilation
import numpy as np



class SegmentationProcessor:
    def __init__(self, image: np.ndarray):
        self._input = image.copy()   # imagem pré-processada, não a original
        self._mask: np.ndarray | None = None
        self._log: list[str] = []

    def apply_threshold_manual(self, value: int = 128, invert: bool = False):
        t = ThresholdManual(value, invert)
        self._mask = t.apply(self._input)
        self._log.append(str(t))

    def apply_threshold_otsu(self, invert: bool = False):
        t = ThresholdOtsu(invert)
        self._mask = t.apply(self._input)
        self._log.append(str(t))

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask.copy() if self._mask is not None else None

    def apply_threshold_adaptive(self, block_size: int = 25, c: float = 5, method: str = "gaussian"):
        t = ThresholdAdaptive(block_size, c, method)
        self._mask = t.apply(self._input)
        self._log.append(str(t))

    def apply_opening(self, kernel_size: int = 3, iterations: int = 1):
        op = Opening(kernel_size, iterations)
        self._mask = op.apply(self._mask)
        self._log.append(str(op))

    def apply_closing(self, kernel_size: int = 3, iterations: int = 1):
        op = Closing(kernel_size, iterations)
        self._mask = op.apply(self._mask)
        self._log.append(str(op))

    def apply_erosion(self, kernel_size: int = 3, iterations: int = 1):
        op = Erosion(kernel_size, iterations)
        self._mask = op.apply(self._mask)
        self._log.append(str(op))

    def apply_dilation(self, kernel_size: int = 3, iterations: int = 1):
        op = Dilation(kernel_size, iterations)
        self._mask = op.apply(self._mask)
        self._log.append(str(op))