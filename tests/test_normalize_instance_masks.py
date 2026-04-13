from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from dataio import projectio
from normalize_instance_masks import normalize_directory


class NormalizeInstanceMasksTests(unittest.TestCase):
    def test_normalize_directory_splits_disconnected_labels_and_relabels_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir(parents=True, exist_ok=True)

            image = np.zeros((5, 5), dtype=np.uint16)
            image[0, 0] = 7
            image[4, 4] = 7
            image[2:4, 1:3] = 9
            tifffile.imwrite(input_dir / "mask000.tif", image)

            stats = normalize_directory(input_dir, output_dir, label_mode="sequential")

            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0].split_label_count, 1)
            self.assertEqual(stats[0].extra_component_count, 1)

            normalized = np.asarray(tifffile.imread(output_dir / "mask000.tif"))
            self.assertEqual(projectio.disconnected_label_components(normalized), ())
            self.assertEqual(sorted(int(value) for value in np.unique(normalized) if int(value) > 0), [1, 2, 3])

    def test_normalize_directory_requires_distinct_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            tifffile.imwrite(directory / "mask000.tif", np.zeros((3, 3), dtype=np.uint16))

            with self.assertRaisesRegex(ValueError, "must be different"):
                normalize_directory(directory, directory)


if __name__ == "__main__":
    unittest.main()
