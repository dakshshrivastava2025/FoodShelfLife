"""Run with: python -m unittest -v test_demo.py"""
import ast
import math
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


def helpers():
    tree = ast.parse(Path(__file__).with_name("demo_app.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_sigmoid", "generate_sensor_reading",
                                   "gas_demo_grade", "build_feature_row",
                                   "capture_webcam_frame"}]
    scope = {"math": math, "np": np, "os": os, "Image": Image,
             "CAMERA_INDEX": next(ast.literal_eval(node.value) for node in tree.body
                                  if isinstance(node, ast.Assign)
                                  and any(isinstance(target, ast.Name)
                                          and target.id == "CAMERA_INDEX"
                                          for target in node.targets))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "demo_helpers", "exec"), scope)
    return scope


class DemoTests(unittest.TestCase):
    def test_scenarios_across_seeds(self):
        funcs = helpers()
        for hours, expected in [(74, "Stale"), (58, "Slightly stale"), (18, "Fresh")]:
            for seed in range(100):
                readings = [funcs["generate_sensor_reading"](i, hours, seed=seed)
                            for i in range(36)]
                a, b = map(list, zip(*readings))
                self.assertEqual(funcs["gas_demo_grade"](a, b), expected)
                self.assertTrue(all(0 <= v <= 1023 for pair in readings for v in pair))

    def test_camera_index_and_backend_retry(self):
        import cv2
        for first_opens in (False, True):
            with self.subTest(first_opens=first_opens):
                first, second = MagicMock(), MagicMock()
                first.isOpened.return_value = first_opens
                first.read.return_value = (False, None)
                second.isOpened.return_value = True
                second.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
                with patch("cv2.VideoCapture", side_effect=[first, second]) as capture:
                    image, status = helpers()["capture_webcam_frame"]()
                self.assertIsNotNone(image)
                self.assertEqual(status, "OK")
                self.assertEqual([call.args for call in capture.call_args_list],
                                 [(1, cv2.CAP_MSMF), (1, cv2.CAP_DSHOW)]
                                 if os.name == "nt" else [(1, cv2.CAP_ANY)])
                first.release.assert_called_once()
                second.release.assert_called_once()

    def test_feature_history_and_clock(self):
        build = helpers()["build_feature_row"]
        features = build(list(range(13)), list(range(20, 33)), 18)[0]
        self.assertEqual(features[4], 12)
        self.assertEqual(features[5], 12)
        self.assertAlmostEqual(features[8], math.sin(2 * math.pi * 5 / 24))
        self.assertEqual(build([70], [250], 18)[0, 4], 0)

    def test_run_complete_reset_and_repeat_without_camera(self):
        camera = MagicMock()
        camera.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=camera):
            app = AppTest.from_file(str(Path(__file__).with_name("demo_app.py"))).run(timeout=40)
            self.assertFalse(app.exception)
            for zone, expected in [("left", "Stale"), ("mid", "Slightly stale"), ("right", "Fresh")]:
                app.button(key=f"hit_{zone}").click().run(timeout=20)
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state.running)
                self.assertTrue(app.button(key="hit_left").disabled)
                # Advance the session clock to test completion without waiting 3 minutes.
                app.session_state.start_time = time.time() - 181
                app.session_state.last_tick_t = time.time() - 181
                app.run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state.done)
                self.assertFalse(app.session_state.running)
                self.assertEqual(app.session_state.tick, 36)
                self.assertEqual(app.session_state.result_name, expected)
                self.assertEqual(app.session_state.camera_status, "Camera not available")
                self.assertAlmostEqual(sum(app.session_state.result_probs), 1)
                next(b for b in app.button if "Reset" in b.label).click().run(timeout=20)
                self.assertFalse(app.exception)
                self.assertFalse(app.session_state.done)
                self.assertEqual(app.session_state.mq135_vals, [])


if __name__ == "__main__":
    unittest.main()
