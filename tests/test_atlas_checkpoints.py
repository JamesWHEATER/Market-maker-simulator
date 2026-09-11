import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import ai_pattern_research_pipeline as ai


class CheckpointTests(unittest.TestCase):
    def test_interrupted_generation_replays_committed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ai.AIExperimentConfig(structures=3)
            specs = ai.sample_run_specs(config)
            kwargs = dict(output_dir=Path(directory), costs=ai.CostModel(),
                          ranges=ai.WorldSamplingRanges(), jobs=1, git_commit=None)
            first = ([{"value": 2.5}], {"simulation_id": 0})
            def interrupted(*args, **kw):
                yield first
                raise RuntimeError("interruption")
            with patch.object(ai, "iter_pattern_runs", side_effect=interrupted):
                iterator = ai.checkpointed_pattern_runs(specs, config, **kwargs)
                self.assertEqual(next(iterator), first)
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    next(iterator)
            tail = [([], {"simulation_id": 1}), ([], {"simulation_id": 2})]
            with patch.object(ai, "iter_pattern_runs", return_value=iter(tail)) as worker:
                result = list(ai.checkpointed_pattern_runs(specs, config, **kwargs))
                self.assertEqual(result, [first] + tail)
                self.assertEqual(worker.call_args.args[0], specs[1:])
            with patch.object(ai, "iter_pattern_runs", return_value=iter(())) as worker:
                self.assertEqual(list(ai.checkpointed_pattern_runs(specs, config, **kwargs)), result)
                self.assertEqual(worker.call_args.args[0], [])
            with self.assertRaisesRegex(ValueError, "settings/code differ"):
                list(ai.checkpointed_pattern_runs(specs, replace(config, base_seed=999), **kwargs))


if __name__ == "__main__":
    unittest.main()
