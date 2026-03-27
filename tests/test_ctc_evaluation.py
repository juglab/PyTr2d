from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tracking.ctc_evaluation import evaluate_result_with_ctc


class CTCEvaluationTests(unittest.TestCase):
    def test_evaluation_skips_when_tool_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_dir = Path(tmpdir) / "02_GT"
            res_dir = Path(tmpdir) / "result"
            gt_dir.mkdir()
            res_dir.mkdir()

            with mock.patch("tracking.ctc_evaluation.find_ctc_evaluate_executable", return_value=None):
                payload = evaluate_result_with_ctc(gt_dir, res_dir)

        self.assertEqual(payload["status"], "skipped")
        self.assertIn("py-ctcmetrics", str(payload["reason"]))

    def test_evaluation_parses_official_csv_and_derives_summary_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_dir = Path(tmpdir) / "02_GT"
            res_dir = Path(tmpdir) / "result"
            gt_dir.mkdir()
            res_dir.mkdir()

            def fake_run(command, capture_output, text, check):
                csv_path = Path(command[command.index("--csv-file") + 1])
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["Valid", "DET", "SEG", "TRA", "LNK", "CT", "TF", "BC(0)", "CCA"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "Valid": "True",
                            "DET": "0.9",
                            "SEG": "0.8",
                            "TRA": "0.7",
                            "LNK": "0.6",
                            "CT": "0.5",
                            "TF": "0.4",
                            "BC(0)": "0.3",
                            "CCA": "0.2",
                        }
                    )
                return mock.Mock(returncode=0, stdout="ok", stderr="")

            with mock.patch("tracking.ctc_evaluation.find_ctc_evaluate_executable", return_value="/tmp/ctc_evaluate"), \
                 mock.patch("tracking.ctc_evaluation.subprocess.run", side_effect=fake_run):
                payload = evaluate_result_with_ctc(gt_dir, res_dir)

        self.assertEqual(payload["status"], "success")
        metrics = payload["metrics"]
        self.assertTrue(metrics["Valid"])
        self.assertAlmostEqual(float(metrics["OP_CSB"]), 0.85)
        self.assertAlmostEqual(float(metrics["OP_CTB"]), 0.75)
        self.assertAlmostEqual(float(metrics["BIO"]), 0.35)
        self.assertAlmostEqual(float(metrics["OP_CLB"]), 0.475)


if __name__ == "__main__":
    unittest.main()
