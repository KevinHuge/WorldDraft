from types import SimpleNamespace

import pytest

from worlddraft_eval.gt.regions import geometry_bounds, project_box


def test_projection_clips_offscreen_geometry():
    assert project_box([(-2, .3, 2), (-1, .8, 2)], 640, 360) is None
    assert project_box([(-.1, .2, 2), (.3, .8, 2)], 640, 360) == pytest.approx([0, 72, 192, 288])


def test_curve_bounds_use_rendered_geometry_and_clear_temporary_mesh():
    cleared = []
    obj = SimpleNamespace(type="CURVE", bound_box=[(-1, -1, -1), (1, 1, 1)],
        to_mesh=lambda: SimpleNamespace(vertices=[SimpleNamespace(co=(-.1, -.1, -.01)),
                                                   SimpleNamespace(co=(.1, .1, .01))]),
        to_mesh_clear=lambda: cleared.append(True))
    bounds = geometry_bounds(obj)
    assert len(bounds) == 8 and max(v[2] for v in bounds) == .01
    assert cleared == [True]
