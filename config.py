from pathlib import Path

from video_processor.app import DEFAULT_CONFIG_PATH
from video_processor.config import load_config

APP_CONFIG = load_config(DEFAULT_CONFIG_PATH)

VIDEO_SOURCE = APP_CONFIG.video.source
OUTPUT_VIDEO = str(APP_CONFIG.video.output)
FPS = APP_CONFIG.video.fps
USE_MEDIAN = APP_CONFIG.filters.use_median
USE_GAUSSIAN = APP_CONFIG.filters.use_gaussian
MEDIAN_KERNEL = APP_CONFIG.filters.median_kernel
GAUSSIAN_KERNEL = APP_CONFIG.filters.gaussian_kernel
ORIGINAL_WINDOW_NAME = "Original Video"
PROCESSED_WINDOW_NAME = APP_CONFIG.window.processed_window_name
CONFIG_PATH = Path(DEFAULT_CONFIG_PATH)
