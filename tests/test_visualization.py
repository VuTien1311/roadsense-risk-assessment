import ast
import sys
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_visualization import ADVISORY_RGB, colorize_depth, make_comparison, render_overlay, save_visualizations


def detection(box, advisory="SLOW", relation="PATH", state="STABLE", unit=4.1, name="car"):
    return {"class": name, "confidence": 0.9, "box_xyxy": list(box), "corridor_relation": relation,
            "original_unit_status": state, "original_inverse_depth_unit": unit, "advisory": advisory, "risk_score": 60.0}


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((360, 640, 3), (60, 60, 60), np.uint8)
        self.road = np.zeros((360, 640), np.uint8)
        self.road[220:, 100:540] = 1
        self.lane = np.zeros((360, 640), np.uint8)
        self.lane[220:, 318:322] = 1

    def test_same_geometry_and_input_unchanged(self):
        before = self.image.copy()
        out = render_overlay(self.image, self.road, self.lane, [detection((200, 120, 320, 220), "BRAKE")], "BRAKE", 0.8,
                             "car PATH 4.1u risk70", {"status": "GOOD", "score": 0.9, "flags": []}, {"status": "CONSISTENT"})
        self.assertEqual((out.shape, out.dtype), (self.image.shape, np.uint8))
        np.testing.assert_array_equal(self.image, before)
        self.assertFalse(np.array_equal(out, before))

    def test_box_color_follows_the_object_advisory(self):
        for advisory in ("MONITOR", "SLOW", "BRAKE", "STOP"):
            out = render_overlay(self.image, None, None, [detection((300, 150, 420, 250), advisory)], advisory)
            red, green, blue = ADVISORY_RGB[advisory]
            edge = out[200, 300:303].astype(int)  # left edge of the box, BGR
            distance = np.abs(edge - np.array([blue, green, red])).sum(axis=1).min()
            self.assertLess(distance, 30, advisory)

    def test_road_tint_only_where_the_mask_is(self):
        out = render_overlay(self.image, self.road, None, [], "CLEAR")
        self.assertGreater(int(out[330, 600, 1]), 0)
        np.testing.assert_array_equal(out[150, 600], self.image[150, 600])  # outside road, card and legend
        self.assertGreater(int(out[300, 320, 1]), int(self.image[300, 320, 1]))  # greener on the road

    def test_no_detection_and_missing_optional_inputs(self):
        out = render_overlay(self.image, None, None, None, "MONITOR")
        self.assertEqual(out.shape, self.image.shape)

    def test_many_objects_border_boxes_and_odd_records(self):
        many = [detection((10 + 24 * i, 100, 30 + 24 * i, 160), "CLEAR", "SIDE", "CHECK") for i in range(25)]
        many += [detection((-20, -10, 5000, 4000), "STOP"), detection((630, 350, 639, 359), "SLOW", unit=None),
                 {"class": "truck"}, {"box_xyxy": [1, 2, 3]}]
        out = render_overlay(self.image, self.road, self.lane, many, "STOP", None, None, {"status": "DEGRADED", "score": 0.1,
                             "flags": ["LOW_LIGHT", "BLUR"]}, {})
        self.assertEqual(out.shape, self.image.shape)

    def test_small_and_large_images_and_mask_of_other_size(self):
        for shape in ((120, 200, 3), (1080, 1920, 3)):
            image = np.zeros(shape, np.uint8)
            out = render_overlay(image, self.road, self.lane, [detection((20, 20, 90, 90))], "SLOW")
            self.assertEqual(out.shape, shape)

    def test_deterministic_and_rejects_bad_images(self):
        args = (self.image, self.road, self.lane, [detection((200, 120, 320, 220))], "SLOW")
        np.testing.assert_array_equal(render_overlay(*args), render_overlay(*args))
        for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((0, 10, 3), np.uint8)):
            with self.assertRaises(ValueError):
                render_overlay(bad)


class ObjectTableTests(unittest.TestCase):
    def test_numbered_overlay_and_table(self):
        from paper_visualization import render_object_table
        image = np.full((360, 640, 3), 60, np.uint8)
        dets = [detection((10 + 20 * i, 100, 28 + 20 * i, 150), "MONITOR", "SIDE") for i in range(30)]
        out = render_overlay(image, None, None, dets, "MONITOR", numbered=True)
        self.assertEqual(out.shape, image.shape)
        table = render_object_table(dets, 360)
        self.assertEqual((table.shape[0], table.shape[2], table.dtype), (360, 3, np.uint8))
        self.assertEqual(render_object_table([], 200).shape[0], 200)


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.original = np.full((180, 320, 3), (40, 70, 100), np.uint8)
        self.overlay = self.original.copy()
        cv2.rectangle(self.overlay, (80, 70), (200, 150), (0, 255, 0), 2)
        self.depth = np.tile(np.linspace(1, 10, 320, dtype=np.float32), (180, 1))

    def test_color_order_and_no_mutation(self):
        before = self.depth.copy()
        view, info = colorize_depth(self.depth)
        np.testing.assert_array_equal(self.depth, before)
        self.assertEqual(view.shape, self.original.shape)
        self.assertGreater(int(view[90, -1, 2]), int(view[90, -1, 0]))
        self.assertGreater(int(view[90, 0, 0]), int(view[90, 0, 2]))
        self.assertEqual(info["valid_fraction"], 1)

    def test_invalid_values(self):
        depth = np.asarray([[np.nan, np.inf, 0, -1]], dtype=np.float32)
        view, info = colorize_depth(depth)
        self.assertEqual(info["valid_fraction"], 0)
        self.assertTrue(np.all(view == 24))

    def test_constant_depth(self):
        view, info = colorize_depth(np.full((12, 16), 5, np.float32))
        self.assertTrue(np.all(view == view[0, 0]))
        self.assertEqual(info["percentile_2"], info["percentile_98"])

    def test_geometry_and_original_unchanged(self):
        view, _ = colorize_depth(self.depth)
        before = self.original.copy()
        result = make_comparison(self.original, self.overlay, view, panel_width=640)
        self.assertEqual(result.shape, (496, 1968, 3))
        np.testing.assert_array_equal(self.original, before)

    def test_bad_shapes_rejected(self):
        view, _ = colorize_depth(self.depth)
        with self.assertRaises(ValueError):
            make_comparison(self.original, self.overlay[:100], view)
        with self.assertRaises(ValueError):
            colorize_depth(np.ones((1, 1, 3)))

    def test_files_and_raw_values_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = save_visualizations(self.original, self.overlay, self.depth, folder, "scene")["files"]
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 0)
            saved = cv2.imread(paths["original"])
            np.testing.assert_array_equal(saved, self.original)
            np.testing.assert_array_equal(np.load(paths["raw_depth"], allow_pickle=False), self.depth)

    def test_same_inference_and_audit_logic(self):
        source = ast.parse((ROOT / "provenance" / "test3_original.py").read_text(encoding="utf-8-sig"))
        updated = ast.parse((ROOT / "test3.py").read_text(encoding="utf-8-sig"))
        self.assertEqual(self.logic(source), self.logic(updated))

    @staticmethod
    def logic(tree):
        # Removing only the new presentation nodes must recover the original AST.
        class StripPresentation(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "paper_visualization" else node

            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id in {"overlay_frame", "visualization"} for target in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_Dict(self, node):
                for index in reversed(range(len(node.keys))):
                    key = node.keys[index]
                    if isinstance(key, ast.Constant) and key.value == "visualization":
                        del node.keys[index]
                        del node.values[index]
                return self.generic_visit(node)
        return ast.dump(StripPresentation().visit(tree), include_attributes=False)


if __name__ == "__main__":
    unittest.main()
