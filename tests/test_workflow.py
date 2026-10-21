import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from railmind import api, augment, data, evaluation, training
from railmind.demo import demo, fixture_png
from railmind.reward import category_reward
from railmind.schema import bbox, checked_path, load, parse_answer, save


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        demo(self.root / "demo")
        self.demo = self.root / "demo"
        self.rows = load(self.demo / "reviewed.jsonl")

    def tearDown(self):
        self.temp.cleanup()

    def test_offline_demo(self):
        self.assertEqual(load(self.demo / "summary.json")["offline_demo"], "PASS")
        self.assertEqual(load(self.demo / "sft/train.json")[0]["messages"][0]["content"].count("<image>"), 1)

    def test_reward_sets(self):
        solution = json.dumps({"anomaly_classes": ["nonut"]})
        cases = [("{\"anomaly_classes\":[\"nonut\"]}", 1),
                 ("{\"anomaly_classes\":[\"nonut\",\"peeling\"]}", .5),
                 ("{\"anomaly_classes\":[]}", 0),
                 ("{\"anomaly_classes\":[\"unknown\"]}", 0),
                 ("not JSON", 0)]
        for answer, expected in cases:
            self.assertEqual(category_reward(answer, solution), expected)

    def test_analysis_prose_is_not_category_prediction(self):
        text = '<think>nonut and peeling are possibilities</think><answer>{"anomaly_classes":[]}</answer>'
        self.assertEqual(parse_answer(text)["anomaly_classes"], [])

    def test_negative_exact_reward(self):
        self.assertEqual(category_reward('{"anomaly_classes":[]}', '{"anomaly_classes":[]}'), 1)

    def test_box_bounds(self):
        for box in ([0, 0, 0, 5], [-1, 0, 1, 2], [0, 0, 50, 50], [0, 0, float("nan"), 1]):
            with self.assertRaises(ValueError):
                bbox(box, 48, 32)

    def test_path_traversal(self):
        with self.assertRaises(ValueError):
            checked_path(self.demo / "images", "../coco.json")

    def test_changed_image_rejected(self):
        fixture_png(self.demo / "images" / self.rows[0]["image"], False)
        with self.assertRaises(ValueError):
            data.verify_manifest(self.rows, self.demo / "images")

    def test_train_split_only(self):
        self.rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            data.export_training(self.rows, self.demo / "images", self.root / "train.json", "sft")

    def test_review_hash_required(self):
        decision = {"id": "defect", "sha256": "wrong", "approved": True, "verified_labels": ["nonut"], "reviewer": "tester"}
        with self.assertRaises(ValueError):
            data.apply_review(self.rows, [decision], self.demo / "images", self.root / "reviewed.jsonl")

    def test_label_correction_requires_corrected_evidence(self):
        decision = {"id": "defect", "sha256": self.rows[0]["sha256"], "approved": True, "verified_labels": [], "reviewer": "tester"}
        with self.assertRaises(ValueError):
            data.apply_review(self.rows, [decision], self.demo / "images", self.root / "reviewed.jsonl")

    def test_pending_samples_not_exported(self):
        for row in self.rows:
            row.pop("review")
        with self.assertRaises(ValueError):
            data.export_training(self.rows, self.demo / "images", self.root / "train.json", "sft")

    def test_rl_real_data_rejected(self):
        with self.assertRaises(ValueError):
            data.export_training(self.rows, self.demo / "images", self.root / "train.jsonl", "rl")

    def test_synthesis_scene_mismatch(self):
        refs = copy.deepcopy([self.rows[0]])
        refs[0]["scene"] = "power_tower"
        with self.assertRaises(ValueError):
            augment.plan([self.rows[1]], refs, [{"labels": ["nonut"], "count": 2}], 42, self.root / "tasks.jsonl")

    def test_synthesis_test_split_rejected(self):
        self.rows[1]["split"] = "test"
        with self.assertRaises(ValueError):
            augment.plan([self.rows[1]], [self.rows[0]], [{"labels": ["nonut"], "count": 2}], 42, self.root / "tasks.jsonl")

    def test_unreviewed_background_rejected(self):
        self.rows[1].pop("review")
        with self.assertRaises(ValueError):
            augment.plan([self.rows[1]], [self.rows[0]], [{"labels": ["nonut"], "count": 2}], 42, self.root / "tasks.jsonl")

    def test_lineage_leakage_rejected(self):
        train = [dict(self.rows[0], source="synthetic", lineage_sha256=[self.rows[1]["sha256"]])]
        test = [dict(self.rows[1], split="test")]
        save(self.root / "train.jsonl", train)
        save(self.root / "test.jsonl", test)
        with self.assertRaises(ValueError):
            data.check_leakage([self.root / "train.jsonl", self.root / "test.jsonl"])

    def test_missing_predictions_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate(self.rows, [], self.root / "metric.json")

    def test_duplicate_boxes_count_as_false_positive(self):
        predictions = load(self.demo / "predictions.jsonl")
        predictions[0]["detections"].append(copy.deepcopy(predictions[0]["detections"][0]))
        metric = evaluation.evaluate(self.rows, predictions, self.root / "metric.json")
        self.assertEqual(metric["detection"]["micro"]["tp"], 1)
        self.assertEqual(metric["detection"]["micro"]["fp"], 1)

    def test_wrong_category_is_not_matched(self):
        predictions = load(self.demo / "predictions.jsonl")
        predictions[0]["detections"][0]["label"] = "peeling"
        metric = evaluation.evaluate(self.rows, predictions, self.root / "metric.json")
        self.assertEqual(metric["detection"]["micro"]["tp"], 0)
        self.assertEqual(metric["detection"]["micro"]["fp"], 1)
        self.assertEqual(metric["detection"]["micro"]["fn"], 1)

    def test_no_gt_in_model_payload(self):
        def response(url, payload, key):
            text = payload["messages"][0]["content"][0]["text"]
            self.assertNotIn("gt_labels", text)
            self.assertNotIn("steel_bridge/defect.png", text)
            return {"choices": [{"message": {"content": '{"scene":"steel_bridge","anomaly_classes":[]}'}}]}
        env = {"RAILMIND_TEACHER_BASE_URL": "https://example.invalid/v1",
               "RAILMIND_TEACHER_API_KEY": "mock-test-only", "RAILMIND_TEACHER_MODEL": "mock"}
        with patch.dict("os.environ", env), patch("railmind.api.request_json", side_effect=response):
            api.annotate(self.demo / "manifest.jsonl", self.demo / "images", Path(__file__).resolve().parents[1] / "prompts/teacher.txt", self.root / "teacher.jsonl", execute=True)
        self.assertNotIn("review", load(self.root / "teacher.jsonl")[0])

    def test_dry_run_never_calls_api(self):
        with patch("railmind.api.request_json") as request:
            result = api.annotate(self.demo / "manifest.jsonl", self.demo / "images", "missing-prompt", self.root / "answer.jsonl")
        request.assert_not_called()
        self.assertTrue(result["dry_run"])

    def test_recipe_count_check(self):
        recipe = Path(__file__).resolve().parents[1] / "configs/sft/server_lora_24265.json"
        with self.assertRaises(ValueError):
            training.launch("sft", recipe, "missing-model", self.demo / "sft/train.json", self.root / "run")

    def test_dino_export_rejects_leakage(self):
        with self.assertRaises(ValueError):
            data.export_dino(self.rows, [dict(self.rows[0], split="val")], self.demo / "images", self.demo / "images", self.root / "dino")

    def test_dino_export_success(self):
        from railmind.schema import digest
        path = self.root / "val/steel_bridge/val.png"
        fixture_png(path, False)
        # 使用不同的示意样例哈希，不代表真实数据集已满足划分独立性。
        path.write_bytes(path.read_bytes() + b"validation-fixture")
        val = [dict(self.rows[0], id="val", image="steel_bridge/val.png", split="val", sha256=digest(path))]
        result = data.export_dino(self.rows, val, self.demo / "images", self.root / "val", self.root / "dino")
        self.assertEqual(result["training_images"], 2)
        coco = load(self.root / "dino/val.coco.json")
        self.assertEqual(coco["annotations"][0]["bbox"], [14.0, 16.0, 8.0, 8.0])

    def test_api_resume_checks_prompt(self):
        prompt = self.root / "prompt.txt"
        prompt.write_text("test prompt")
        env = {"RAILMIND_TEACHER_BASE_URL": "https://example.invalid/v1",
               "RAILMIND_TEACHER_API_KEY": "mock-test-only", "RAILMIND_TEACHER_MODEL": "mock"}
        response = {"choices": [{"message": {"content": '{"anomaly_classes":[]}'}}]}
        with patch.dict("os.environ", env), patch("railmind.api.request_json", return_value=response):
            api.annotate(self.demo / "manifest.jsonl", self.demo / "images", prompt, self.root / "teacher.jsonl", execute=True)
            prompt.write_text("changed prompt")
            with self.assertRaises(ValueError):
                api.annotate(self.demo / "manifest.jsonl", self.demo / "images", prompt, self.root / "teacher.jsonl", execute=True)

    def test_nano_resume_keeps_accepted_task_id(self):
        tasks = load(self.demo / "augmentation_tasks.jsonl")[:1]
        prompts = self.root / "prompts.json"
        save(prompts, {"category_mapping": {"nonut": "fixture prompt"}})
        submissions = []
        def response(url, payload, key, **kwargs):
            if url.endswith("/nano-banana"):
                submissions.append(payload)
                return {"code": 0, "data": {"id": "accepted-task"}}
            self.assertEqual(payload["id"], "accepted-task")
            raise RuntimeError("simulated network interruption during poll")
        env = {"NANO_BANANA_BASE_URL": "https://example.invalid", "NANO_BANANA_API_KEY": "mock-test-only"}
        with patch.dict("os.environ", env), patch("railmind.augment.request_json", side_effect=response):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    augment.run(tasks, self.demo / "images", self.demo / "images", prompts, self.root / "generated", execute=True)
        self.assertEqual(len(submissions), 1)

    def test_explicit_subset(self):
        from railmind.schema import subset_manifest
        result = subset_manifest(self.rows, ["defect"], self.root / "subset.jsonl")
        self.assertEqual(result["images"], 1)
        self.assertEqual(load(self.root / "subset.jsonl")[0]["id"], "defect")


if __name__ == "__main__":
    unittest.main()
