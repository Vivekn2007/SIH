from .base import SchedulerPolicy


class OpenLoopScheduler(SchedulerPolicy):
    """Fixed-dwell baseline integration point."""

    # >>> PLACEHOLDER — I WILL REPLACE THIS WITH MY OWN OPEN-LOOP ALGORITHM <<<
    def __init__(self, band_count: int = 36) -> None:
        self.band_count = band_count
        self.current_band = -1

    def reset(self) -> None:
        self.current_band = -1

    def select_band(self, observation: dict) -> int:
        self.current_band = (self.current_band + 1) % self.band_count
        return self.current_band
