"""Camera projection and reference-video timing checks; no rendering or GPU."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from worlddraft_eval.gt.camera import export_case, intrinsics_from_projection


class CameraGTTests(unittest.TestCase):
    def test_pixel_intrinsics_preserve_letterbox_and_noncentral_principal_point(self):
        fx = 42 / 36 * 1280
        projection = [[2 * fx / 1280, 0, .2, 0], [0, 2 * fx / 720, -.1, 0],
                      [0, 0, -1, -1], [0, 0, -1, 0]]
        K, transform = intrinsics_from_projection(projection, [1280, 720], [640, 360], [832, 480])
        self.assertEqual(transform["resized_size"], [832, 468])
        self.assertEqual(transform["offset_xy"], [0, 6])
        self.assertAlmostEqual(K["fx"], 42 / 36 * 832)
        self.assertAlmostEqual(K["fy"], K["fx"])
        self.assertAlmostEqual(K["cx"], .4 * 832)
        self.assertAlmostEqual(K["cy"], .45 * 468 + 6)

    @unittest.skipUnless(importlib.util.find_spec("bpy"), "Run integration check in the bpy-gt environment")
    def test_real_d01_export_matches_video_timeline_and_blender_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            gt = export_case("D01", Path(directory))
            self.assertTrue((Path(directory) / "gt/D01.json").exists())
        frames = gt["camera_trajectory"]
        self.assertEqual((gt["fps"], gt["frame_count"]), (16, 81))
        self.assertEqual([r["frame"] for r in frames], list(range(81)))
        self.assertEqual((frames[0]["blender_frame"], frames[-1]["blender_frame"]), (1, 148))
        self.assertEqual(frames[-1]["time_s"], 5)
        self.assertEqual(frames[1]["source_frame"], frames[2]["source_frame"])
        self.assertEqual(frames[1]["position"], frames[2]["position"])
        self.assertEqual(frames[1]["quaternion"], frames[2]["quaternion"])
        K = frames[0]["intrinsics"]
        self.assertAlmostEqual(K["fx"], 42 / 36 * 832, places=3)
        self.assertAlmostEqual(K["fy"], K["fx"], places=3)
        self.assertEqual((K["cx"], K["cy"]), (416, 240))
        self.assertTrue(all(abs(math.sqrt(sum(x*x for x in r["quaternion"])) - 1) < 1e-6 for r in frames))
        self.assertLess(gt["validation"]["max_projection_error_px"], .01)
        self.assertEqual(gt["evaluation_grid"]["last_time_s"], 4.875)


if __name__ == "__main__":
    unittest.main()
