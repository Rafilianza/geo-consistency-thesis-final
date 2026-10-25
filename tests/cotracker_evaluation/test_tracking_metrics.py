"""Synthetic metric contracts; these do not validate CoTracker GPU accuracy."""

import ast
import json
from pathlib import Path
import unittest

import numpy as np

from cotracker_evaluation import tracking_metrics
from cotracker_evaluation.tracking_metrics import CLASS_NAMES, build_reverse_queries, compute_metrics, select_queries


def stationary_case(frame_count=81, anchors=(0,), height=12, width=20):
    labels = np.full((frame_count, height, width), 2, dtype=np.uint8)
    queries = np.asarray([(anchor, 4, 4) for anchor in anchors], dtype=np.float32).reshape(-1, 3)
    classes = np.full(len(queries), 2, dtype=np.int64)
    tracks = np.broadcast_to(queries[None, :, 1:], (frame_count, len(queries), 2)).copy()
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    return labels, tracks, visibility, queries, classes


class QuerySelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_distributed_and_uses_existing_pixels(self):
        labels = np.zeros((81, 10, 12), dtype=np.uint8)
        labels[:, :, 6:] = 2
        labels[:, 5, 5] = 255
        first_queries, first_classes = select_queries(labels)
        second_queries, second_classes = select_queries(labels)
        np.testing.assert_array_equal(first_queries, second_queries)
        np.testing.assert_array_equal(first_classes, second_classes)
        self.assertEqual(first_queries.dtype, np.float32)
        self.assertEqual(first_classes.dtype, np.int64)
        self.assertEqual(first_queries.shape, (3 * 2 * 16, 3))
        self.assertEqual(set(first_classes), {0, 2})
        for anchor in (0, 32, 64):
            for class_id in (0, 2):
                group = first_queries[(first_queries[:, 0] == anchor) & (first_classes == class_id)]
                self.assertEqual(len(np.unique(group, axis=0)), 16)
                self.assertGreater(float(np.ptp(group[:, 2])), 7)
        for query, class_id in zip(first_queries, first_classes):
            frame, column, row = query.astype(int)
            self.assertEqual(labels[frame, row, column], class_id)

    def test_rare_class_and_short_video_do_not_invent_queries(self):
        labels = np.full((2, 3, 3), 255, dtype=np.uint8)
        labels[0, 1, 1] = 1
        queries, classes = select_queries(labels)
        np.testing.assert_array_equal(queries, [[0, 1, 1]])
        np.testing.assert_array_equal(classes, [1])

    def test_empty_selection_has_expected_shapes(self):
        labels = np.full((81, 2, 3), 255, dtype=np.uint8)
        queries, classes = select_queries(labels)
        self.assertEqual(queries.shape, (0, 3))
        self.assertEqual(classes.shape, (0,))
        self.assertEqual(select_queries(np.zeros_like(labels), points_per_class=0)[0].shape, (0, 3))


class TrackingMetricsTests(unittest.TestCase):
    def test_identity_has_perfect_agreement_zero_motion_and_independent_cycle(self):
        labels, tracks, visibility, queries, classes = stationary_case(anchors=(0, 32, 64))
        reverse_queries, indices = build_reverse_queries(tracks, visibility, queries, 20, 12)
        np.testing.assert_array_equal(reverse_queries[:, 0], [64, 32, 0])
        reverse_tracks = tracks[::-1, indices]
        result = compute_metrics(labels, tracks, visibility, queries, classes,
                                 reverse_tracks, visibility[::-1, indices], indices)
        row = result["per_class"][2]
        self.assertEqual(row["class_agreement"], 1)
        self.assertEqual(row["original_class_persistence"], 1)
        self.assertEqual(row["motion_diagonals_per_second_median"], 0)
        self.assertEqual(row["cycle_error_pixel_median"], 0)
        self.assertEqual(row["cycle_error_diagonal_p90"], 0)
        self.assertEqual(row["cycle_eligible_count"], 3)
        self.assertEqual(row["cycle_valid_return_count"], 3)
        self.assertEqual(row["cycle_valid_return_fraction"], 1)
        self.assertEqual(row["retention_16"], 1)
        self.assertEqual(row["continuous_retention_16"], 1)
        self.assertEqual(row["valid_observation_count"], 80 + 48 + 16)
        self.assertEqual(row["valid_pair_count"], 79 + 47 + 15)
        json.dumps(result, allow_nan=False)

    def test_translation_is_not_penalized_and_motion_uses_fps_and_diagonal(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=20, width=100)
        tracks[:, 0, 0] += np.arange(20)
        _, indices = build_reverse_queries(tracks, visibility, queries, 100, 12)
        result = compute_metrics(labels, tracks, visibility, queries, classes,
                                 tracks[::-1], visibility[::-1], indices, fps=10)
        row = result["per_class"][2]
        self.assertEqual(row["class_agreement"], 1)
        self.assertEqual(row["original_class_persistence"], 1)
        self.assertAlmostEqual(row["motion_diagonals_per_second_median"], 10 / np.hypot(100, 12))
        self.assertEqual(row["cycle_error_pixel_median"], 0)

    def test_perspective_growth_and_accelerating_motion_are_legitimate(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=20, anchors=(0, 0), height=100, width=100)
        queries[:, 1:] = [[35, 35], [65, 65]]
        scale = 1 + np.square(np.arange(20) / 30)
        tracks[:] = 50 + (queries[None, :, 1:] - 50) * scale[:, None, None]
        _, indices = build_reverse_queries(tracks, visibility, queries, 100, 100)
        result = compute_metrics(labels, tracks, visibility, queries, classes,
                                 tracks[::-1], visibility[::-1], indices)
        row = result["per_class"][2]
        self.assertGreater(np.linalg.norm(tracks[-1, 0] - tracks[-1, 1]),
                           np.linalg.norm(tracks[0, 0] - tracks[0, 1]))
        self.assertEqual(row["class_agreement"], 1)
        self.assertEqual(row["original_class_persistence"], 1)
        self.assertEqual(row["cycle_error_diagonal_median"], 0)
        self.assertGreater(row["motion_diagonals_per_second_median"], 0)

    def test_forced_query_frame_does_not_hide_permanent_class_switch(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=6)
        labels[1:] = 3
        row = compute_metrics(labels, tracks, visibility, queries, classes)["per_class"][2]
        self.assertEqual(row["class_agreement"], 1)
        self.assertEqual(row["original_class_persistence"], 0)
        self.assertEqual(row["valid_observation_count"], 5)
        self.assertEqual(row["valid_pair_count"], 4)

    def test_forced_query_visibility_is_excluded_from_continuous_retention(self):
        labels, tracks, visibility, queries, classes = stationary_case()
        visibility[0] = False
        row = compute_metrics(labels, tracks, visibility, queries, classes)["per_class"][2]
        self.assertEqual(row["continuous_retention_16"], 1)
        self.assertEqual(row["original_class_persistence"], 1)

    def test_nan_out_of_bounds_occlusion_and_unknown_labels_are_excluded(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=7)
        visibility[2] = False
        tracks[3] = np.nan
        tracks[4, 0] = [20, 4]
        labels[5, 4, 4] = 255
        result = compute_metrics(labels, tracks, visibility, queries, classes,
                                 cycle_horizon=4, retention_horizons=(2, 4, 6))
        row = result["per_class"][2]
        self.assertEqual(row["valid_observation_count"], 2)
        self.assertEqual(row["valid_pair_count"], 0)
        self.assertIsNone(row["class_agreement"])
        self.assertEqual(row["original_class_persistence"], 1)
        self.assertEqual(row["retention_2"], 0)
        self.assertEqual(row["retention_4"], 0)
        self.assertEqual(row["retention_6"], 1)
        self.assertEqual(row["continuous_retention_6"], 0)
        self.assertEqual(row["cycle_forward_endpoint_count"], 0)
        self.assertEqual(result["per_query"][0]["motion_pair_count"], 1)
        json.dumps(result, allow_nan=False)

    def test_no_future_observations_produce_unavailable_not_perfect_scores(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=3, anchors=(2,))
        row = compute_metrics(labels, tracks, visibility, queries, classes)["per_class"][2]
        self.assertEqual(row["query_count"], 1)
        self.assertEqual(row["valid_observation_count"], 0)
        self.assertEqual(row["retention_8_eligible_count"], 0)
        for name in ("class_agreement", "original_class_persistence", "retention_8",
                     "continuous_retention_8", "cycle_error_pixel_median", "motion_diagonals_per_second_median"):
            self.assertIsNone(row[name], name)

    def test_query_first_average_does_not_weight_long_tracks_more(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=6, anchors=(0, 0))
        queries[1, 1:] = [8, 4]
        tracks[:, 1] = [8, 4]
        labels[2:, 4, 8] = 3
        visibility[3:, 1] = False
        row = compute_metrics(labels, tracks, visibility, queries, classes)["per_class"][2]
        self.assertEqual(row["valid_pair_count"], 5)
        self.assertEqual(row["class_agreement_query_count"], 2)
        self.assertEqual(row["class_agreement"], 0.5)
        self.assertEqual(row["original_class_persistence"], 0.75)

    def test_motion_percentiles_summarize_query_medians(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=6, anchors=(0, 0), width=100)
        tracks[:, 0, 0] += np.arange(6)
        tracks[:, 1, 0] += 3 * np.arange(6)
        visibility[3:, 1] = False
        row = compute_metrics(labels, tracks, visibility, queries, classes, fps=1)["per_class"][2]
        self.assertAlmostEqual(row["motion_diagonals_per_second_median"], 2 / np.hypot(100, 12))
        self.assertAlmostEqual(row["motion_diagonals_per_second_p90"], 2.8 / np.hypot(100, 12))

    def test_nearest_neighbor_labels_use_pixel_centers(self):
        labels, tracks, visibility, queries, classes = stationary_case(frame_count=3)
        labels[1:, 4, 5] = 3
        tracks[1:, 0] = [4.5, 4]
        row = compute_metrics(labels, tracks, visibility, queries, classes)["per_class"][2]
        self.assertEqual(row["original_class_persistence"], 0)
        self.assertEqual(row["class_agreement"], 1)

    def test_all_missing_classes_and_zero_queries_remain_in_output(self):
        labels, tracks, visibility, queries, classes = stationary_case(anchors=())
        result = compute_metrics(labels, tracks, visibility, queries, classes)
        self.assertEqual(len(result["per_class"]), 11)
        self.assertEqual(result["per_query"], [])
        self.assertEqual(result["per_class"][2]["class_pixel_fraction"], 1)
        for class_id, row in enumerate(result["per_class"]):
            self.assertEqual(row["class_name"], CLASS_NAMES[class_id])
            self.assertEqual(row["query_count"], 0)
            self.assertIsNone(row["class_agreement"])
            if class_id != 2:
                self.assertEqual(row["class_pixel_fraction"], 0)

    def test_visibility_confidence_is_not_silently_thresholded(self):
        labels, tracks, visibility, queries, classes = stationary_case()
        with self.assertRaisesRegex(ValueError, "not confidence"):
            compute_metrics(labels, tracks, visibility.astype(float) * 0.9, queries, classes)


class ReverseCycleTests(unittest.TestCase):
    def test_reverse_queries_use_fixed_endpoint_and_full_video_time(self):
        labels, tracks, visibility, queries, classes = stationary_case(anchors=(0, 32, 64, 65))
        tracks[16, 0] = [7, 8]
        visibility[48, 1] = False
        reverse_queries, indices = build_reverse_queries(tracks, visibility, queries, 20, 12)
        np.testing.assert_array_equal(indices, [0, 2])
        np.testing.assert_array_equal(reverse_queries, [[64, 7, 8], [0, 4, 4]])

    def test_return_is_measured_at_original_anchor_not_reverse_query_time(self):
        labels, tracks, visibility, queries, classes = stationary_case(anchors=(32,))
        reverse_queries, indices = build_reverse_queries(tracks, visibility, queries, 20, 12)
        reverse_tracks = tracks[::-1].copy()
        reverse_tracks[48, 0] = [7, 8]  # Return error 5 pixels at T-1-anchor.
        reverse_tracks[32, 0] = reverse_queries[0, 1:]  # Forced endpoint is perfect.
        row = compute_metrics(labels, tracks, visibility, queries, classes,
                              reverse_tracks, visibility[::-1], indices)["per_class"][2]
        self.assertEqual(row["cycle_error_pixel_median"], 5)
        self.assertAlmostEqual(row["cycle_error_diagonal_median"], 5 / np.hypot(20, 12))

    def test_endpoint_and_return_coverage_keep_hidden_tracks_in_denominators(self):
        labels, tracks, visibility, queries, classes = stationary_case(anchors=(0, 0, 0))
        visibility[16, 2] = False
        _, indices = build_reverse_queries(tracks, visibility, queries, 20, 12)
        reverse_tracks = tracks[::-1, indices].copy()
        reverse_visibility = visibility[::-1, indices].copy()
        reverse_visibility[80, 1] = False
        row = compute_metrics(labels, tracks, visibility, queries, classes,
                              reverse_tracks, reverse_visibility, indices)["per_class"][2]
        self.assertEqual(row["cycle_eligible_count"], 3)
        self.assertEqual(row["cycle_forward_endpoint_count"], 2)
        self.assertEqual(row["cycle_reverse_query_count"], 2)
        self.assertEqual(row["cycle_valid_return_count"], 1)
        self.assertAlmostEqual(row["cycle_forward_endpoint_fraction"], 2 / 3)
        self.assertEqual(row["cycle_valid_return_fraction"], 0.5)
        self.assertAlmostEqual(row["retention_16"], 2 / 3)

    def test_missing_reverse_inference_is_unavailable(self):
        labels, tracks, visibility, queries, classes = stationary_case()
        row = compute_metrics(labels, tracks, visibility, queries, classes)["per_class"][2]
        self.assertEqual(row["cycle_forward_endpoint_count"], 1)
        self.assertIsNone(row["cycle_valid_return_fraction"])
        self.assertIsNone(row["cycle_error_pixel_median"])


class CompatibilityTests(unittest.TestCase):
    def test_module_parses_as_python39(self):
        source = Path(tracking_metrics.__file__)
        ast.parse(source.read_text(), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
