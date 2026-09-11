"""Storage contracts exercised with cheap simulation results, not huge fixtures."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pandas as pd

import ai_pattern_research_pipeline as ai
import atlas_streaming_data as streaming


def fake_runs(specs, experiment, **kwargs):
    for spec in specs:
        run_id = ai.simulator_run_id(spec.config)
        count = 0 if spec.simulation_id % 4 == 0 else 1 + spec.simulation_id % 5
        rows = [{
            "run_id": run_id, "structure_id": spec.structure_id,
            "event_id": f"event-{spec.simulation_id}-{index}",
            "profitable": index % 2, "net_directional_return": 0.01 if index % 2 else -0.01,
            "entry_mid": 100.0, "exit_mid": 101.0,
            "feature": float(index), "missing": float("nan"),
        } for index in range(count)]
        yield rows, {"simulation_id": spec.simulation_id, "run_id": run_id,
                     "structure_id": spec.structure_id, "world_family": spec.world_family,
                     "training_rows": count, "skipped_incomplete_horizon": 1}


class StreamingDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.experiment = ai.AIExperimentConfig(structures=9, seeds_per_structure=2,
                                                steps_per_run=100, null_world_fraction=0.333)

    def generate(self, name="data", **kwargs):
        with mock.patch.object(ai, "iter_pattern_runs", side_effect=fake_runs), \
                mock.patch.object(ai, "sample_run_specs", side_effect=AssertionError("full design allocation")):
            return streaming.generate_streaming_dataset(self.root / name, self.experiment, **kwargs)

    def test_bounded_batches_structure_splits_and_empty_runs(self):
        dataset = self.generate(runs_per_shard=4)
        manifest = dataset.manifest
        self.assertEqual(manifest["generated_runs"], 18)
        self.assertEqual(manifest["generated_structures"], 9)
        self.assertEqual(manifest["shards"], 5)
        actual_read_csv = pd.read_csv
        sizes = []

        def bounded_read(*args, **kwargs):
            sizes.append(kwargs.get("chunksize"))
            self.assertGreater(kwargs.get("chunksize", 0), 0)
            return actual_read_csv(*args, **kwargs)

        with mock.patch.object(pd, "read_csv", side_effect=bounded_read):
            batches = list(dataset.iter_batches(batch_rows=2))
            self.assertTrue(all(0 < len(frame) <= 2 for frame in batches))
            runs = list(dataset.iter_runs())
        self.assertTrue(sizes)
        self.assertEqual(len(runs), 18)
        self.assertEqual(sum(len(frame) for frame, _ in runs), manifest["rows"])
        self.assertEqual([summary["simulation_id"] for _, summary in runs], list(range(18)))
        self.assertTrue(runs[0][0].empty)
        self.assertEqual(list(runs[0][0].columns), manifest["columns"])
        for index in range(0, 18, 2):
            first, second = runs[index][1], runs[index + 1][1]
            self.assertEqual(first["structure_id"], second["structure_id"])
            self.assertEqual(first["split"], second["split"])
            self.assertNotEqual(first["seed"], second["seed"])
        all_rows = pd.concat(batches, ignore_index=True)
        self.assertTrue(all_rows.groupby("structure_id")["split"].nunique().eq(1).all())
        for split in streaming.PARTITIONS:
            selected = list(dataset.iter_batches(split, batch_rows=3, columns=["event_id"]))
            self.assertTrue(all(list(frame.columns) == ["event_id"] for frame in selected))
            self.assertEqual(sum(len(frame) for frame in selected), manifest["rows_by_split"][split])
            selected_runs = list(dataset.iter_runs(split))
            self.assertTrue(all(summary["split"] == split for _, summary in selected_runs))
            self.assertEqual(len(selected_runs), manifest["runs_by_split"][split])
        with closing(sqlite3.connect(dataset.root / streaming.CATALOG_FILENAME)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM structures WHERE world_family LIKE 'null_%'").fetchone()[0], 3)
            self.assertEqual(connection.execute("SELECT COUNT(DISTINCT seed) FROM runs").fetchone()[0], 18)

    def test_design_and_rows_are_invariant_to_shard_and_reader_size(self):
        first = self.generate("first", runs_per_shard=2)
        second = self.generate("second", runs_per_shard=8)
        frames = [pd.concat(list(dataset.iter_batches(batch_rows=size)), ignore_index=True)
                  for dataset, size in ((first, 2), (second, 17))]
        pd.testing.assert_frame_equal(*frames)
        self.assertEqual(first.manifest["dataset_id"], second.manifest["dataset_id"])
        self.assertEqual([summary for _, summary in first.iter_runs()],
                         [summary for _, summary in second.iter_runs()])

    def test_resume_rolls_back_partial_shard_without_repeating_committed_runs(self):
        output = self.root / "interrupted"

        def interrupted(specs, experiment, **kwargs):
            for result in fake_runs(specs, experiment):
                if result[1]["simulation_id"] == 5:
                    raise RuntimeError("power interruption")
                yield result

        with mock.patch.object(ai, "iter_pattern_runs", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "power interruption"):
                streaming.generate_streaming_dataset(output, self.experiment, runs_per_shard=4)
        manifest = json.loads((output / streaming.MANIFEST_FILENAME).read_text())
        self.assertEqual(manifest["status"], "FAILED")
        self.assertEqual(manifest["generated_runs"], 4)
        with self.assertRaisesRegex(ValueError, "not a complete"):
            streaming.StreamingDataset(output)
        observed = []

        def resumed(specs, experiment, **kwargs):
            observed.extend(spec.simulation_id for spec in specs)
            yield from fake_runs(specs, experiment)

        with mock.patch.object(ai, "iter_pattern_runs", side_effect=resumed):
            actual = streaming.generate_streaming_dataset(output, self.experiment, runs_per_shard=6, resume=True)
        self.assertEqual(observed, list(range(4, 18)))
        expected = self.generate("uninterrupted", runs_per_shard=2)
        pd.testing.assert_frame_equal(
            pd.concat(list(actual.iter_batches()), ignore_index=True),
            pd.concat(list(expected.iter_batches()), ignore_index=True),
        )
        actual.verify_integrity()

    def test_complete_resume_does_not_simulate_and_mismatched_config_rejected(self):
        dataset = self.generate()
        with mock.patch.object(ai, "iter_pattern_runs", side_effect=AssertionError("must not rerun")):
            resumed = streaming.generate_streaming_dataset(dataset.root, self.experiment, resume=True)
        self.assertEqual(dataset.manifest, resumed.manifest)
        with self.assertRaisesRegex(ValueError, "settings/code differ"):
            streaming.generate_streaming_dataset(dataset.root, replace(self.experiment, base_seed=43), resume=True)
        self.assertEqual(dataset.manifest["status"], "COMPLETE")

    def test_shard_catalog_and_manifest_tampering_are_rejected(self):
        shard_data = self.generate("shard")
        shard = next((shard_data.root / "shards").glob("*.csv.gz"))
        with shard.open("ab") as handle:
            handle.write(b"tampering")
        with self.assertRaisesRegex(ValueError, "shard changed"):
            list(shard_data.iter_batches())
        with self.assertRaisesRegex(ValueError, "shard changed"):
            streaming.generate_streaming_dataset(shard_data.root, self.experiment, resume=True)
        self.assertEqual(shard_data.manifest["status"], "COMPLETE")
        catalog_data = self.generate("catalog")
        with closing(sqlite3.connect(catalog_data.root / streaming.CATALOG_FILENAME)) as connection:
            connection.execute("UPDATE runs SET row_count=999 WHERE simulation_id=0")
            connection.commit()
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            streaming.StreamingDataset(catalog_data.root)
        self.assertEqual(catalog_data.manifest["status"], "COMPLETE")
        manifest_data = self.generate("manifest")
        manifest = manifest_data.manifest
        manifest["rows"] += 1
        (manifest_data.root / streaming.MANIFEST_FILENAME).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            streaming.StreamingDataset(manifest_data.root)

    def test_million_structure_request_only_samples_first_bounded_shard(self):
        experiment = replace(self.experiment, structures=1_000_000)
        lengths = []

        def stop_after_first_batch(specs, experiment, **kwargs):
            lengths.append(len(specs))
            raise RuntimeError("stop before expensive simulation")

        with mock.patch.object(ai, "iter_pattern_runs", side_effect=stop_after_first_batch), \
                mock.patch.object(ai, "sample_run_specs", side_effect=AssertionError("full specs")):
            with self.assertRaisesRegex(RuntimeError, "stop before"):
                streaming.generate_streaming_dataset(self.root / "million", experiment, runs_per_shard=4)
        self.assertEqual(lengths, [4])
        with closing(sqlite3.connect(self.root / "million" / streaming.CATALOG_FILENAME)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM structures").fetchone()[0], 0)

    def test_all_zero_event_runs_remain_readable(self):
        def no_events(specs, experiment, **kwargs):
            for _, summary in fake_runs(specs, experiment):
                summary["training_rows"] = 0
                yield [], summary

        with mock.patch.object(ai, "iter_pattern_runs", side_effect=no_events):
            dataset = streaming.generate_streaming_dataset(self.root / "empty", self.experiment, runs_per_shard=4)
        self.assertEqual(dataset.manifest["rows"], 0)
        self.assertEqual(list(dataset.iter_batches()), [])
        self.assertEqual(len(list(dataset.iter_runs())), 18)

    def test_omitted_simulation_cannot_commit_or_claim_complete(self):
        with mock.patch.object(ai, "iter_pattern_runs", return_value=iter(())):
            with self.assertRaisesRegex(ValueError, "omitted planned runs"):
                streaming.generate_streaming_dataset(self.root / "missing", self.experiment)
        manifest = json.loads((self.root / "missing" / streaming.MANIFEST_FILENAME).read_text())
        self.assertEqual(manifest["generated_runs"], 0)
        self.assertEqual(manifest["status"], "FAILED")

    def test_small_generation_reports_100_percent_only_after_frozen_manifest(self):
        lines = []
        output = self.root / "progress"

        def capture(line, **kwargs):
            lines.append(line)
            manifest = json.loads((output / streaming.MANIFEST_FILENAME).read_text())
            self.assertEqual(manifest["status"], "COMPLETE")

        with mock.patch("builtins.print", side_effect=capture):
            self.generate("progress", runs_per_shard=4)
        self.assertEqual(len(lines), 1)
        self.assertIn("18/18 runs (100.00%)", lines[0])
        self.assertIn("ETA 00:00:00", lines[0])

    def test_failed_generation_does_not_print_final_completion(self):
        def interrupted(specs, experiment, **kwargs):
            for result in fake_runs(specs, experiment):
                if result[1]["simulation_id"] == 5:
                    raise RuntimeError("interrupted")
                yield result

        with mock.patch.object(ai, "iter_pattern_runs", side_effect=interrupted), mock.patch("builtins.print") as output:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                streaming.generate_streaming_dataset(self.root / "progress_failure", self.experiment, runs_per_shard=4)
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
