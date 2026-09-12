from abc import ABC, abstractmethod


class SchedulerPolicy(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear per-run scheduler state."""

    @abstractmethod
    def select_band(self, observation: dict) -> int:
        """Return the band ID (0-35) to dwell on for this timestep."""
