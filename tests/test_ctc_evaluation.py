from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tracking.ctc_evaluation import evaluate_result_with_ctc, find_ctc_evaluate_executable


class CTCEvaluationTests(unittest.TestCase):
    def test_find_ctc_evaluate_executable_checks_unresolved_python_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_env = Path(tmpdir) / ".venv"
            fake_bin = fake_env / "bin"
            fake_bin.mkdir(parents=True)
            fake_python = fake_bin / "python"
            fake_ctc = fake_bin / "ctc_evaluate"
            fake_python.write_text("", encoding="utf-8")
            fake_ctc.write_text("", encoding="utf-8")

            with mock.patch("tracking.ctc_evaluation.shutil.which", return_value=None), \
                 mock.patch("tracking.ctc_evaluation.sys.executable", str(fake_python)), \
                 mock.patch("tracking.ctc_evaluation.sys.prefix", str(fake_env)), \
                 mock.patch("pathlib.Path.resolve", autospec=True, side_effect=lambda path: Path("/nonexistent/python")):
                executable = find_ctc_evaluate_executable()

        self.assertEqual(executable, str(fake_ctc))

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

    def test_evaluation_falls_back_to_gt_res_track_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_dir = Path(tmpdir) / "02_GT"
            res_dir = Path(tmpdir) / "result"
            (gt_dir / "TRA").mkdir(parents=True)
            res_dir.mkdir()
            (gt_dir / "TRA" / "res_track.txt").write_text("1 0 0 0\n", encoding="utf-8")

            def fake_run(command, capture_output, text, check):
                effective_gt_dir = Path(command[command.index("--gt") + 1])
                man_track_path = effective_gt_dir / "TRA" / "man_track.txt"
                self.assertTrue(man_track_path.exists())
                self.assertEqual(man_track_path.read_text(encoding="utf-8"), "1 0 0 0\n")

                csv_path = Path(command[command.index("--csv-file") + 1])
                csv_path.write_text("Valid;TRA\nTrue;0.7\n", encoding="utf-8")
                return mock.Mock(returncode=0, stdout="ok", stderr="")

            with mock.patch("tracking.ctc_evaluation.find_ctc_evaluate_executable", return_value="/tmp/ctc_evaluate"), \
                 mock.patch("tracking.ctc_evaluation.subprocess.run", side_effect=fake_run):
                payload = evaluate_result_with_ctc(gt_dir, res_dir)

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["gt_lineage_fallback"], "TRA/res_track.txt")
        self.assertTrue(payload["metrics"]["Valid"])
        self.assertAlmostEqual(float(payload["metrics"]["TRA"]), 0.7)

    def test_evaluation_parses_semicolon_delimited_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_dir = Path(tmpdir) / "02_GT"
            res_dir = Path(tmpdir) / "result"
            gt_dir.mkdir()
            res_dir.mkdir()

            def fake_run(command, capture_output, text, check):
                csv_path = Path(command[command.index("--csv-file") + 1])
                csv_path.write_text(
                    "Valid;DET;SEG;TRA;LNK;CT;TF;BC(0);CCA\n"
                    "1;0.9;0.8;0.7;0.6;0.5;0.4;0.3;0.2\n",
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0, stdout="ok", stderr="")

            with mock.patch("tracking.ctc_evaluation.find_ctc_evaluate_executable", return_value="/tmp/ctc_evaluate"), \
                 mock.patch("tracking.ctc_evaluation.subprocess.run", side_effect=fake_run):
                payload = evaluate_result_with_ctc(gt_dir, res_dir)

        self.assertEqual(payload["status"], "success")
        metrics = payload["metrics"]
        self.assertEqual(metrics["Valid"], 1)
        self.assertAlmostEqual(float(metrics["TRA"]), 0.7)
        self.assertAlmostEqual(float(metrics["OP_CTB"]), 0.75)


if __name__ == "__main__":
    unittest.main()
