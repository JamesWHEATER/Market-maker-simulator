import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ai_pattern_research_pipeline as ai
import ai_pattern_dataset_generator as generator
import atlas_pipeline as atlas


class StreamingCLITests(unittest.TestCase):
    def test_existing_entry_point_defaults_to_disk_streaming(self):
        args = ai.build_argument_parser().parse_args(["all", "--structures", "1000000", "--seeds-per-structure", "2"])
        self.assertFalse(args.legacy_in_memory)
        self.assertEqual(args.output_dir.drive.upper(), "D:")
        with mock.patch.object(atlas, "run_streaming_cli") as streamed, \
                mock.patch.object(ai, "build_training_dataset", side_effect=AssertionError("in-memory generation")):
            ai._run_cli_impl(args)
            streamed.assert_called_once_with(args)

    def test_invalid_rerun_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "atlas_status.json"
            path.write_text(json.dumps({"status": "EVALUATION_COMPLETE", "confirmatory_results_valid": True}))
            args = ai.build_argument_parser().parse_args(["train", "--output-dir", str(root), "--jobs", "-1"])
            with mock.patch.object(atlas, "require_d_destination", return_value=root):
                with self.assertRaisesRegex(ValueError, "jobs 1 or 2"):
                    atlas.run_streaming_cli(args)
            self.assertEqual(json.loads(path.read_text())["status"], "FAILED")
            self.assertFalse(json.loads(path.read_text())["confirmatory_results_valid"])

    def test_dataset_entry_point_dispatches_without_full_csv_generator(self):
        argv = ["ai_pattern_dataset_generator.py", "--structures", "1000000", "--seeds-per-structure", "2"]
        with mock.patch("sys.argv", argv), mock.patch.object(atlas, "run_streaming_cli") as streamed, \
                mock.patch.object(generator, "generate_dataset", side_effect=AssertionError("legacy generator")):
            generator.main()
            self.assertEqual(streamed.call_args.args[0].command, "build-dataset")
            self.assertEqual(streamed.call_args.args[0].structures, 1000000)

    def test_rejected_destination_creates_no_directory_or_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must_not_be_created"
            args = ai.build_argument_parser().parse_args(["build-dataset", "--output-dir", str(output)])
            with mock.patch.object(atlas, "require_d_destination", side_effect=ValueError("must be D:")):
                with self.assertRaisesRegex(ValueError, "D:"):
                    ai.run_cli(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
