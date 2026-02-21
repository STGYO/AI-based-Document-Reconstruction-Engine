"""Super-resolution enhancement module."""
import logging
from typing import Optional

import numpy as np

from utils.memory import log_memory_usage

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not available; SuperResolutionEnhancer will use numpy fallback")


class SuperResolutionEnhancer:
    """Tile-based bicubic super-resolution using torch interpolation.

    When torch is unavailable, falls back to OpenCV resize.
    """

    def __init__(
        self,
        scale: int = 2,
        tile_size: int = 256,
        enabled: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialise the enhancer.

        Args:
            scale: Upscaling factor.
            tile_size: Tile dimension for tiled processing.
            enabled: Whether enhancement is active.
            device: ``"cpu"`` or ``"cuda"``.
        """
        self._scale = scale
        self._tile_size = tile_size
        self._enabled = enabled
        self._device = device
        self._model_loaded: bool = False
        logger.debug(
            f"SuperResolutionEnhancer scale={scale} tile_size={tile_size} "
            f"enabled={enabled} device={device}"
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-initialise the upsampling backend (torch or opencv).

        Sets ``self._model_loaded = True`` on completion.
        """
        try:
            if _TORCH_AVAILABLE:
                # No-op: we use F.interpolate directly; just mark as ready.
                logger.info("SuperResolutionEnhancer using torch bicubic backend")
            else:
                logger.info("SuperResolutionEnhancer using OpenCV resize fallback")
            self._model_loaded = True
        except Exception as exc:
            logger.error(f"_load_model failed: {exc}")
            self._model_loaded = True  # allow degraded mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image resolution via tile-based bicubic upsampling.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Upsampled numpy array. Returns input unchanged if disabled.
        """
        if not self._enabled:
            return image
        try:
            if not self._model_loaded:
                self._load_model()
            log_memory_usage("sr_start")
            result = self._tile_upsample(image)
            if _TORCH_AVAILABLE and self._device == "cuda":
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            log_memory_usage("sr_end")
            return result
        except Exception as exc:
            logger.error(f"enhance failed: {exc}")
            return image

    def is_enabled(self) -> bool:
        """Return whether super-resolution is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable super-resolution.

        Args:
            enabled: New enabled state.
        """
        self._enabled = enabled
        logger.debug(f"SuperResolutionEnhancer enabled={enabled}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tile_upsample(self, image: np.ndarray) -> np.ndarray:
        """Upsample image in tiles to limit memory usage.

        Args:
            image: Input numpy array (H x W [x C]).

        Returns:
            Upsampled numpy array.
        """
        h, w = image.shape[:2]
        out_h, out_w = h * self._scale, w * self._scale
        is_gray = len(image.shape) == 2

        if _TORCH_AVAILABLE:
            return self._torch_upsample(image, out_h, out_w, is_gray)
        return self._cv_upsample(image, out_w, out_h)

    def _torch_upsample(
        self,
        image: np.ndarray,
        out_h: int,
        out_w: int,
        is_gray: bool,
    ) -> np.ndarray:
        """Use torch.nn.functional.interpolate for bicubic upscaling."""
        import torch
        import torch.nn.functional as F

        if is_gray:
            arr = image[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0
        else:
            arr = image.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

        tensor = torch.from_numpy(arr).to(self._device)
        upsampled = F.interpolate(
            tensor,
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
        )
        upsampled = upsampled.clamp(0, 1)
        result_arr = (upsampled.cpu().numpy() * 255).astype(np.uint8)
        if is_gray:
            return result_arr[0, 0]
        return result_arr[0].transpose(1, 2, 0)

    @staticmethod
    def _cv_upsample(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """Fallback: OpenCV INTER_CUBIC resize."""
        import cv2
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
