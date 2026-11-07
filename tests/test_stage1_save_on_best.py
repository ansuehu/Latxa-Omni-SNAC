import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers import TrainerControl

from omni_speech.train.stage1 import SaveOnBestDevCallback


class SaveOnBestDevTest(unittest.TestCase):
    def run_steps(self, output_dir, callback, steps, best_metric=None, best_checkpoint=None):
        args = SimpleNamespace(output_dir=output_dir, save_only_model=False)
        state = SimpleNamespace(
            global_step=0, is_world_process_zero=True,
            best_metric=best_metric, best_model_checkpoint=best_checkpoint,
        )
        callback.on_train_begin(args, state, TrainerControl())
        weights_only = {}
        for global_step, loss, epoch_end in steps:
            state.global_step = global_step
            control = TrainerControl()
            if epoch_end:
                callback.on_epoch_end(args, state, control)
                control.should_save = True
            if loss is not None:
                callback.on_evaluate(args, state, control, metrics={"eval_loss": loss})
            if control.should_save:
                checkpoint = Path(output_dir) / f"checkpoint-{global_step}"
                checkpoint.mkdir()
                if not args.save_only_model:
                    (checkpoint / f"global_step{global_step}").mkdir()
                weights_only[global_step] = args.save_only_model
                callback.on_save(args, state, control)
        kept = sorted(int(p.name.split("-")[1]) for p in Path(output_dir).iterdir())
        return kept, weights_only

    def test_keeps_latest_best_and_epoch_checkpoints(self):
        with tempfile.TemporaryDirectory() as output_dir:
            kept, weights_only = self.run_steps(output_dir, SaveOnBestDevCallback(), [
                (250, 2.0, False),
                (500, 1.5, False),   # new best: removes 250
                (600, None, True),   # epoch save, not a best
                (750, 1.6, False),   # worse: no save
                (1000, 1.2, True),   # best at epoch end: full save, removes 500
                (1250, 1.1, False),  # new best: keeps 1000 (epoch)
            ])
            self.assertEqual(kept, [600, 1000, 1250])
            self.assertEqual(weights_only, {
                250: True, 500: True, 600: False, 1000: False, 1250: True,
            })

    def test_resume_continues_from_saved_best(self):
        with tempfile.TemporaryDirectory() as output_dir:
            old_best = Path(output_dir) / "checkpoint-2750"
            (old_best / "global_step2750").mkdir(parents=True)
            kept, weights_only = self.run_steps(
                output_dir, SaveOnBestDevCallback(), [
                    (3000, 0.79, False),   # worse than the restored best: no save
                    (3250, 0.77, False),   # new best; old full checkpoint is kept
                    (3500, 0.76, False),   # new best: removes 3250
                ],
                best_metric=0.78, best_checkpoint=str(old_best),
            )
            self.assertEqual(kept, [2750, 3500])
            self.assertEqual(weights_only, {3250: True, 3500: True})

    def test_patience_stops_after_evals_without_new_best(self):
        args = SimpleNamespace(output_dir="unused", save_only_model=False)
        state = SimpleNamespace(global_step=0, best_metric=None, best_model_checkpoint=None)
        callback = SaveOnBestDevCallback(patience=2)
        callback.on_train_begin(args, state, TrainerControl())
        stopped = []
        for global_step, loss in [(250, 1.0), (500, 1.1), (750, 0.9), (1000, 0.95), (1250, 0.92)]:
            state.global_step = global_step
            control = callback.on_evaluate(args, state, TrainerControl(), metrics={"eval_loss": loss})
            stopped.append(control.should_training_stop)
        # 750 resets the count; 1000 and 1250 are the two evals without improvement.
        self.assertEqual(stopped, [False, False, False, False, True])

    def test_no_patience_never_stops(self):
        args = SimpleNamespace(output_dir="unused", save_only_model=False)
        state = SimpleNamespace(global_step=0, best_metric=None, best_model_checkpoint=None)
        callback = SaveOnBestDevCallback()
        callback.on_train_begin(args, state, TrainerControl())
        for global_step, loss in [(250, 1.0)] + [(250 * i, 2.0) for i in range(2, 12)]:
            state.global_step = global_step
            control = callback.on_evaluate(args, state, TrainerControl(), metrics={"eval_loss": loss})
            self.assertFalse(control.should_training_stop)


if __name__ == "__main__":
    unittest.main()
