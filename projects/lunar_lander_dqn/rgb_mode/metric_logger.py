from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Union


class MetricLogger:
    """
    Minimal TensorBoard logger for tracking metrics like loss, reward, epsilon, etc.
    """

    def __init__(self, log_dir: Union[str, Path] = "runs", experiment_name: Optional[str] = None):
        """
        Args:
            log_dir: Base directory where TensorBoard logs will be stored.
            experiment_name: Optional experiment subfolder name.
        """
        log_path = Path(log_dir)
        if experiment_name:
            log_path = log_path / experiment_name
        log_path.mkdir(parents=True, exist_ok=True)

        self.__writer = SummaryWriter(log_dir=str(log_path))

    def log(self, tag: str, value: float, step: int) -> None:
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag: Name of the metric (e.g., "loss", "reward", "epsilon").
            value: Metric value.
            step: Step number (e.g., training iteration or episode index).
        """
        self.__writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """Closes the logger."""
        self.__writer.close()
