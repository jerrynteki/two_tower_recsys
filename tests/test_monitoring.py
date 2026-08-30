import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from training.monitoring import log_configuration, log_validation_metrics


class MonitoringTests(unittest.TestCase):
    def test_tensorboard_writes_configuration_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = SummaryWriter(directory)
            log_configuration(writer, {"embedding_dim": 64})
            log_validation_metrics(writer, {"Recall@10": 0.25}, step=1)
            writer.close()

            events = event_accumulator.EventAccumulator(str(Path(directory)))
            events.Reload()

            self.assertIn("validation/recall_at_10", events.Tags()["scalars"])
            self.assertEqual(
                events.Scalars("validation/recall_at_10")[0].value,
                0.25,
            )


if __name__ == "__main__":
    unittest.main()
