import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from evaluation.evaluate import load_model
from training import train
from training.checkpointing import capture_random_state, restore_random_state, save_checkpoint


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def run_training(self, arguments):
        with (
            patch("sys.argv", ["training.train", "--no-tensorboard", *arguments]),
            patch.object(train, "select_device", return_value=torch.device("cpu")),
            patch.object(train, "evaluate_retrieval_metrics", return_value={"NDCG@1": 1.0}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train.main()

    def test_interrupted_training_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"user_idx": [0, 1, 0, 1], "movie_idx": [0, 1, 2, 3]}).to_csv(
                root / "train.csv", index=False
            )
            pd.DataFrame({"user_idx": [0, 1], "movie_idx": [3, 2]}).to_csv(
                root / "val.csv", index=False
            )
            for filename, count in (("user2idx.json", 2), ("movie2idx.json", 7)):
                (root / filename).write_text(json.dumps({str(i): i for i in range(count)}))
            common = [
                "--processed-dir", str(root), "--epochs", "8", "--batch-size", "2",
                "--embedding-dim", "4", "--negative-count", "2", "--seed", "7",
                "--eval-every", "2", "--patience", "2", "--ks", "1",
                "--selection-metric", "NDCG@1",
            ]
            full = root / "full.pt"
            self.run_training([*common, "--output", str(full)])
            expected = torch.load(root / "full.latest.pt", weights_only=True)
            self.assertEqual(expected["epoch"], 6)
            self.assertEqual(expected["early_stopping"]["best_epoch"], 2)
            self.assertTrue(expected["early_stopping"]["stopped"])
            self.assertTrue(any(
                not torch.equal(value, expected["best_model_state_dict"][name])
                for name, value in expected["model_state_dict"].items()
            ))

            # Exercise interruption before any evaluation and with a partially
            # exhausted patience counter, preserving the original epoch budget.
            for interrupted_epoch in (1, 4):
                with self.subTest(interrupted_epoch=interrupted_epoch):
                    output = root / f"split{interrupted_epoch}.pt"
                    latest = output.with_suffix(".latest.pt")

                    def interrupt(checkpoint, path):
                        save_checkpoint(checkpoint, path)
                        if checkpoint.get("epoch") == interrupted_epoch:
                            raise InterruptedError("simulated process interruption")

                    with patch.object(train, "save_checkpoint", side_effect=interrupt):
                        with self.assertRaises(InterruptedError):
                            self.run_training([*common, "--output", str(output)])
                    saved = torch.load(latest, weights_only=True)
                    self.assertEqual(saved["epoch"], interrupted_epoch)
                    if interrupted_epoch == 1:
                        self.assertIsNone(saved["best_model_state_dict"])
                    else:
                        self.assertEqual(saved["early_stopping"]["evaluations_without_improvement"], 1)
                    self.run_training([
                        "--resume", str(latest), "--epochs", "8", "--output", str(output)
                    ])
                    self.assert_nested_equal(expected, torch.load(latest, weights_only=True))
                    self.assert_nested_equal(
                        torch.load(full, weights_only=True), torch.load(output, weights_only=True)
                    )
                    # Best-model exports remain loadable by the retrieval API.
                    load_model(output, torch.device("cpu"))
                    with patch.object(train, "train_sampled_epoch", side_effect=AssertionError("must stay stopped")):
                        self.run_training([
                            "--resume", str(latest), "--epochs", "8", "--output", str(output)
                        ])
                    self.run_training([
                        "--resume", str(latest), "--epochs", "7", "--output", str(output),
                        "--no-early-stopping",
                    ])
                    self.assertEqual(torch.load(latest, weights_only=True)["epoch"], 7)

    def test_random_states_round_trip_through_safe_loader(self):
        loader = torch.Generator().manual_seed(17)
        sampler = torch.Generator().manual_seed(23)

        def draws():
            values = [random.random(), np.random.rand(), torch.rand(3),
                      torch.rand(3, generator=loader), torch.rand(3, generator=sampler)]
            if torch.cuda.is_available():
                values.extend(torch.rand(3, device=f"cuda:{index}").cpu()
                              for index in range(torch.cuda.device_count()))
            if torch.backends.mps.is_available():
                values.append(torch.rand(3, device="mps").cpu())
            return values

        # Exercise NumPy's cached Gaussian state as well as its generator array.
        np.random.normal()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random.pt"
            save_checkpoint(capture_random_state(loader, sampler), path)
            expected = draws()
            restore_random_state(torch.load(path, weights_only=True), loader, sampler)
            self.assert_nested_equal(expected, draws())

    def test_failed_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint({"epoch": 1}, path)
            with patch("training.checkpointing.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True), {"epoch": 1})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_best_model_export_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint({"model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "resumable latest checkpoint"):
                self.run_training(["--resume", str(path)])


if __name__ == "__main__":
    unittest.main()
