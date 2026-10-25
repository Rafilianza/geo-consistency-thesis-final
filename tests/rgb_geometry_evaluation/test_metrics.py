"""Small CPU contracts; never load models, CUDA, or historical result data."""

import json
import unittest
from unittest import mock

import numpy as np

from cotracker_evaluation.tracking_metrics import build_reverse_queries, select_queries
from rgb_geometry_evaluation import metrics, track


def point_fixture(class_id=2, anchors=(0,)):
    labels = np.full((81, 9, 11), class_id, dtype=np.uint8)
    queries = np.array([[anchor, 5, 4] for anchor in anchors], dtype=np.float32)
    tracks = np.repeat(queries[None, :, 1:], 81, axis=0)
    visible = np.ones((81, len(queries)), dtype=bool)
    classes = np.full(len(queries), class_id, dtype=np.int64)
    return labels, tracks, visible, queries, classes


class PersistenceTests(unittest.TestCase):
    def test_every_genuine_class_gets_the_same_lifetime_formula(self):
        for class_id in metrics.SCORED_CLASS_IDS:
            with self.subTest(class_id=class_id):
                labels, tracks, visible, queries, classes = point_fixture(class_id)
                labels[41:] = 255
                row = metrics.class_persistence(labels, tracks, visible, queries, classes)[class_id]
                self.assertEqual(row["score"], 0.5)
                self.assertEqual(row["visible_observation_count"], 80)
                self.assertEqual(row["unknown_observation_count"], 40)
                self.assertEqual(row["per_query"][0]["observed_label_counts"]["unknown"], 40)

    def test_unknown_zero_and_255_are_mismatches_not_valid_background(self):
        labels, tracks, visible, queries, classes = point_fixture()
        labels[1:21] = 0
        labels[21:41] = 255
        rows = metrics.class_persistence(labels, tracks, visible, queries, classes)
        self.assertEqual(rows[2]["score"], 0.5)
        self.assertEqual(rows[2]["unknown_observation_count"], 40)
        self.assertIsNone(rows[0]["score"])
        self.assertFalse(rows[0]["supported"])
        self.assertEqual(rows[0]["support_reason"], "background_is_unknown")

    def test_occlusion_reentry_and_class_change_do_not_end_scoring(self):
        labels, tracks, visible, queries, classes = point_fixture()
        visible[20:30] = False
        labels[20:30] = 9
        labels[40] = 9
        row = metrics.class_persistence(labels, tracks, visible, queries, classes)[2]
        self.assertEqual(row["score"], 69 / 70)
        self.assertEqual(row["per_query"][0]["last_observed_frame"], 80)
        self.assertEqual(row["per_query"][0]["observed_label_counts"]["vegetation"], 1)
        self.assertEqual(row["per_query"][0]["continuous_visible_seconds"], 19 / 16)
        self.assertEqual(row["per_query"][0]["observable_seconds"], 70 / 16)

    def test_invalid_positions_are_unobserved(self):
        labels, tracks, visible, queries, classes = point_fixture()
        tracks[1, 0] = np.nan
        tracks[2, 0] = np.inf
        tracks[3, 0] = [-1, 4]
        tracks[4, 0] = [11, 4]
        visible[5, 0] = False
        row = metrics.class_persistence(labels, tracks, visible, queries, classes)[2]
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["visible_observation_count"], 75)
        self.assertEqual(row["censored_observation_count"], 5)
        self.assertEqual(row["invisible_observation_count"], 1)
        self.assertEqual(row["nonfinite_observation_count"], 2)
        self.assertEqual(row["out_of_frame_observation_count"], 2)

    def test_unobserved_point_keeps_equal_weight(self):
        labels, tracks, visible, queries, classes = point_fixture(anchors=(0, 64))
        visible[1:, 0] = False
        row = metrics.class_persistence(labels, tracks, visible, queries, classes)[2]
        self.assertEqual(row["score"], 0.5)
        self.assertEqual(row["zero_observation_queries"], 1)
        self.assertEqual(row["availability"], 2 / 80)
        self.assertEqual(row["by_anchor"]["0"]["score"], 0)
        self.assertEqual(row["by_anchor"]["64"]["score"], 1)

    def test_five_anchors_have_five_different_possible_durations(self):
        labels, tracks, visible, queries, classes = point_fixture(anchors=metrics.ANCHORS)
        row = metrics.class_persistence(labels, tracks, visible, queries, classes)[2]
        self.assertEqual([point["possible_seconds"] for point in row["per_query"]], [5, 4, 3, 2, 1])
        self.assertEqual([point["visible_observation_count"] for point in row["per_query"]], [80, 64, 48, 32, 16])
        self.assertEqual(set(row["by_anchor"]), {"0", "16", "32", "48", "64"})

    def test_empty_classes_remain_zero_with_explicit_missing_support(self):
        labels = np.full((81, 9, 11), 255, dtype=np.uint8)
        queries, classes = metrics.select_evaluation_queries(labels)
        rows = metrics.class_persistence(labels, np.empty((81, 0, 2)),
                                         np.empty((81, 0), dtype=bool), queries, classes)
        self.assertEqual(len(rows), 11)
        self.assertIsNone(rows[0]["score"])
        for row in rows[1:]:
            self.assertEqual(row["score"], 0)
            self.assertFalse(row["supported"])
            self.assertEqual(row["support_reason"], "no_class_queries")
            self.assertIsNone(row["retention_16"])
            self.assertIsNone(row["cycle_valid_return_fraction"])
        json.dumps(rows, allow_nan=False)

    def test_class_at_start_must_match_and_background_cannot_seed(self):
        labels, tracks, visible, queries, classes = point_fixture()
        with self.assertRaises(ValueError):
            metrics.class_persistence(labels, tracks, visible, queries, np.array([1]))
        with self.assertRaises(ValueError):
            metrics.class_persistence(labels, tracks, visible, queries, np.array([0]))

    def test_reverse_error_is_paired_with_return_coverage(self):
        labels, tracks, visible, queries, classes = point_fixture(anchors=(0, 16))
        reverse_queries, indices = build_reverse_queries(tracks, visible, queries, 11, 9, 16)
        reverse_tracks = np.repeat(reverse_queries[None, :, 1:], 81, axis=0)
        reverse_visible = np.ones((81, 2), dtype=bool)
        reverse_tracks[80, 0, 0] += 1
        reverse_visible[64, 1] = False
        row = metrics.class_persistence(labels, tracks, visible, queries, classes,
                                        reverse_tracks, reverse_visible, indices)[2]
        self.assertEqual(row["cycle_error_pixel_median"], 1.0)
        self.assertEqual(row["cycle_valid_return_fraction"], 0.5)
        self.assertEqual(row["cycle_valid_return_count"], 1)
        self.assertEqual(row["cycle_forward_endpoint_count"], 2)
        self.assertEqual(row["retention_16"], 1)
        self.assertEqual(row["by_anchor"]["0"]["cycle_valid_return_fraction"], 1)
        self.assertEqual(row["by_anchor"]["16"]["cycle_valid_return_fraction"], 0)

    def test_fully_unobserved_class_is_not_supported(self):
        labels, tracks, visible, queries, classes = point_fixture()
        visible[1:] = False
        row = metrics.class_persistence(labels, tracks, visible, queries, classes)[2]
        self.assertEqual(row["score"], 0)
        self.assertFalse(row["supported"])
        self.assertEqual(row["support_reason"], "no_later_visible_observations")
        self.assertEqual(row["zero_observation_queries"], 1)


class SelectionAndMotionTests(unittest.TestCase):
    def test_selection_preserves_other_classes_and_erodes_cars_only(self):
        labels = np.full((81, 15, 21), 6, dtype=np.uint8)
        labels[:, 3:12, 5:16] = 2
        labels[:, 1:3, 1:3] = 2
        labels[:, 0] = 0
        original = labels.copy()
        untouched_queries, untouched_classes = select_queries(labels, metrics.ANCHORS)
        queries, classes = metrics.select_evaluation_queries(labels)
        np.testing.assert_array_equal(labels, original)
        np.testing.assert_array_equal(queries[classes == 6], untouched_queries[untouched_classes == 6])
        self.assertFalse(np.any(classes == 0))
        car_queries = queries[classes == 2]
        self.assertEqual(len(car_queries), 80)
        self.assertEqual(set(car_queries[:, 0]), set(metrics.ANCHORS))
        self.assertTrue(((car_queries[:, 1] >= 7) & (car_queries[:, 1] <= 13)).all())
        self.assertTrue(((car_queries[:, 2] >= 5) & (car_queries[:, 2] <= 9)).all())

    def test_background_motion_accepts_new_anchors_without_cars(self):
        labels, tracks, visible, queries, _ = point_fixture(6, metrics.ANCHORS)
        result = metrics.background_motion(labels, tracks, visible, queries)
        self.assertTrue(result["supported"])
        self.assertEqual(result["supported_segments"], 5)
        self.assertEqual(result["gross_displacement"], 0)

    def test_photo_windows_cover_every_later_frame_once(self):
        frames = np.zeros((81, 9, 11, 3), dtype=np.uint8)
        positions = [[1, 1], [5, 1], [9, 1], [1, 7], [5, 7], [9, 7]]
        queries = np.array([[anchor, x, y] for anchor in metrics.ANCHORS for x, y in positions], dtype=np.float32)
        tracks = np.repeat(queries[None, :, 1:], 81, axis=0)
        visible = np.ones((81, len(queries)), dtype=bool)
        result = metrics.one_second_photo_diagnostic(frames, tracks, visible, queries)
        self.assertTrue(result["supported"])
        self.assertEqual(result["mean_brightness_aligned_rgb_mae"], 0)
        self.assertEqual([row["frame_index"] for row in result["frames"]], list(range(1, 81)))
        self.assertEqual([window["query_count"] for window in result["windows"]], [6] * 5)

    def test_analyze_road_only_still_keeps_motion_and_all_class_rows(self):
        labels = np.full((81, 9, 11), 6, dtype=np.uint8)
        frames = np.zeros((*labels.shape, 3), dtype=np.uint8)

        def fake_track(predictor, rgb, queries, device):
            return (np.repeat(queries[None, :, 1:], len(rgb), axis=0),
                    np.ones((len(rgb), len(queries)), dtype=bool))

        with mock.patch.object(track, "track_video", side_effect=fake_track) as mocked:
            arrays, result = track.analyze_tracks(None, frames, labels, device="cpu")
        self.assertEqual(mocked.call_count, 3)
        self.assertTrue(result["background_motion"]["supported"])
        self.assertTrue(result["animated_photo"]["supported"])
        self.assertEqual(result["per_class"][2]["query_count"], 0)
        self.assertEqual(result["per_class"][6]["score"], 1)
        self.assertEqual(len(arrays["queries"]), 80)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
