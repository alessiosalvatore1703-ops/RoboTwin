"""Pins the base-offset contract: off is None, offsets are rigid about the nominal base, draws are paired."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "base_offset", Path(__file__).resolve().parents[1] / "envs" / "utils" / "base_offset.py")
base_offset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base_offset)

KWARGS = {"dual_arm_embodied": True, "left_embodiment_config": {"robot_pose": [[0, -0.65, 0, 0.707, 0, 0, 0.707]]}}


def test_no_base_knob_resolves_to_none_and_no_matrix():
    assert base_offset.resolve_base_offset({"task_name": "lift_pot"}, seed=3) is None
    assert base_offset.base_offset_matrix(None, KWARGS) is None


def test_zero_offset_matrix_is_identity():
    offset = base_offset.resolve_base_offset({"base_dx": 0, "base_dy": 0, "base_dyaw": 0}, seed=0)
    np.testing.assert_allclose(base_offset.base_offset_matrix(offset, KWARGS), np.eye(4), atol=1e-12)


def test_yaw_pivots_on_nominal_base_origin():
    matrix = base_offset.base_offset_matrix({"dx": 0.1, "dy": 0.0, "dyaw": np.pi / 2}, KWARGS)
    pivot = np.array([0, -0.65, 0, 1])
    np.testing.assert_allclose(matrix @ pivot, [0.1, -0.65, 0, 1], atol=1e-12)
    nominal = np.eye(4)
    nominal[:2, :2] = [[0, -1], [1, 0]]  # base heading +y
    nominal[:3, 3] = [0, -0.65, 0]
    x, y, yaw = base_offset.pose_xy_yaw(matrix @ nominal)
    assert (x, y) == pytest.approx((0.1, -0.65)) and yaw == pytest.approx(np.pi)


def test_random_offsets_are_paired_by_seed_and_bounded():
    args = {"base_noise_xy_m": 0.2, "base_noise_yaw_rad": 0.4, "noise_seed": 7}
    first = base_offset.resolve_base_offset(args, seed=100000)
    assert first == base_offset.resolve_base_offset(dict(args), seed=100000)
    assert first != base_offset.resolve_base_offset(args, seed=100001)
    draws = [base_offset.resolve_base_offset(args, seed=s) for s in range(200)]
    assert max(abs(d["dx"]) for d in draws) <= 0.2 and max(abs(d["dyaw"]) for d in draws) <= 0.4


def test_fixed_and_random_knobs_are_rejected_together():
    with pytest.raises(ValueError):
        base_offset.resolve_base_offset({"base_dx": 0.1, "base_noise_xy_m": 0.1}, seed=0)


def test_offset_rejected_for_non_dual_arm_embodiment():
    with pytest.raises(ValueError):
        base_offset.base_offset_matrix({"dx": 0, "dy": 0, "dyaw": 0}, {**KWARGS, "dual_arm_embodied": False})


class _Entity:
    def __init__(self, name):
        self.name = name


class _Body:
    def __init__(self, entity):
        self.entity = entity


class _Contact:
    def __init__(self, first, second, separation):
        self.bodies = [_Body(first), _Body(second)]
        self.points = [type("P", (), {"separation": separation})()]


class _Scene:
    def __init__(self, contacts):
        self._contacts = contacts

    def get_contacts(self):
        return self._contacts


LINK = _Entity("fl_link3")


def _penetrations(first, second, separation):
    scene = _Scene([_Contact(first, second, separation)])
    return base_offset.robot_penetrations(scene, [type("L", (), {"entity": LINK})()])


def test_penetration_flags_deep_robot_scene_contact_only():
    assert _penetrations(LINK, _Entity("wall"), -0.05) == [("fl_link3", "wall", pytest.approx(0.05))]
    assert _penetrations(_Entity("wall"), LINK, -0.05) == [("fl_link3", "wall", pytest.approx(0.05))]


def test_penetration_ignores_floor_touching_and_non_robot_pairs():
    assert _penetrations(LINK, _Entity("ground"), -0.05) == []
    assert _penetrations(LINK, _Entity("table"), -5e-5) == []
    assert _penetrations(LINK, _Entity("table"), 0.01) == []
    assert _penetrations(_Entity("pot"), _Entity("table"), -0.05) == []
