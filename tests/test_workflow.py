"""Integration checks using the downloaded NASA data and selected checkpoint."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_server import app, service
from data_loader import ALL_DATASETS, load_data, prepare_test_samples, risk_category
from download_data import EXPECTED


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_nasa_terminal_labels_and_short_histories(self):
        for dataset in ALL_DATASETS:
            with self.subTest(dataset=dataset):
                _, test, rul = load_data(dataset)
                windows, targets, ids = prepare_test_samples(
                    test, rul, service.config["feature_columns"], service.seq_length, max_rul=None
                )
                self.assertEqual(len(ids), EXPECTED[dataset][-1])
                np.testing.assert_array_equal(targets, rul.RUL.to_numpy()[ids - 1])
                short_id = test.groupby("unit_id").size().idxmin()
                short = test[test.unit_id == short_id].sort_values("cycle")
                window = windows[np.flatnonzero(ids == short_id)[0]]
                expected = short[service.config["feature_columns"]].to_numpy(dtype=np.float32)
                np.testing.assert_array_equal(window[-min(len(expected), service.seq_length):], expected[-service.seq_length:])
                if len(expected) < service.seq_length:
                    padding = service.seq_length - len(expected)
                    np.testing.assert_array_equal(window[:padding], np.repeat(expected[:1], padding, axis=0))

    def test_api_matches_offline_metrics_and_raw_sequence(self):
        self.assertTrue(service.available)
        self.assertTrue(service.evaluation)
        summary = self.client.get("/api/summary")
        self.assertEqual(summary.status_code, 200)
        for dataset in ALL_DATASETS:
            with self.subTest(dataset=dataset):
                response = self.client.get(f"/api/evaluate-fresh?dataset={dataset}")
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                offline = service.evaluation["datasets"][dataset]
                self.assertEqual(data["engineCount"], offline["engines"])
                self.assertAlmostEqual(data["rmse"], offline["capped"]["rmse"], places=3)
                self.assertEqual(data["rulCap"], service.max_rul)

        _, test, rul = load_data("FD004")
        windows, _, ids = prepare_test_samples(
            test, rul, service.config["feature_columns"], service.seq_length, max_rul=None
        )
        sample = self.client.get(f"/api/sample-prediction?dataset=FD004&engineId={ids[0]}").get_json()
        payload = {"dataset": "FD004", "sequence": windows[0].tolist()}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        inference = response.get_json()
        self.assertAlmostEqual(inference["predictedRul"], sample["predictedRul"], places=4)
        interval = inference["predictionInterval"]
        self.assertEqual(interval["target"], 0.90)
        self.assertEqual(interval["method"], "validation-scaled")
        self.assertGreater(interval["upper"], interval["lower"])
        self.assertGreaterEqual(interval["lower"], 0)
        self.assertEqual(self.client.post("/api/predict", json={**payload, "mcSamples": 0}).status_code, 400)

    def test_risk_thresholds(self):
        self.assertEqual(
            risk_category(np.array([30, 30.01, 60, 60.01])).tolist(),
            ["Critical", "Warning", "Warning", "Healthy"],
        )


if __name__ == "__main__":
    unittest.main()
