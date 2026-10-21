"""Base-pose perturbation: a rigid world-frame move of the robot (and its head camera) against a fixed scene.

Substrate layer, numpy only so it is testable without SAPIEN. Offsets are (dx, dy) in metres and dyaw in
radians, all in the world frame, with yaw taken about the nominal base origin.
"""
import numpy as np

# Contact separations are numerically noisy at touching distance (|sep| of order 1e-5 m on a valid pose),
# so only penetration deeper than this counts as an invalid base pose.
PENETRATION_TOL_M = 1e-3
IGNORED_CONTACT_NAMES = ("ground",)

FIXED_KEYS = ("base_dx", "base_dy", "base_dyaw")
NOISE_KEYS = ("base_noise_xy_m", "base_noise_yaw_rad")


def resolve_base_offset(usr_args, seed):
    """Per-episode offset dict from eval args, or None when no base knob is set (byte-identical off).

    Random offsets are a pure function of (noise_seed, episode seed) so every policy sees the same
    perturbation for the same episode seed: the comparison is paired.
    """
    fixed = [key for key in FIXED_KEYS if key in usr_args]
    noise = [key for key in NOISE_KEYS if key in usr_args]
    if fixed and noise:
        raise ValueError(f"fixed {fixed} and random {noise} base offsets are mutually exclusive")
    if fixed:
        return {"dx": float(usr_args.get("base_dx", 0.0)), "dy": float(usr_args.get("base_dy", 0.0)),
                "dyaw": float(usr_args.get("base_dyaw", 0.0))}
    if noise:
        xy_m = float(usr_args.get("base_noise_xy_m", 0.0))
        yaw_rad = float(usr_args.get("base_noise_yaw_rad", 0.0))
        if not (np.isfinite(xy_m) and np.isfinite(yaw_rad) and xy_m >= 0 and yaw_rad >= 0):
            raise ValueError("base noise ranges must be finite and nonnegative")
        rng = np.random.default_rng([int(usr_args.get("noise_seed", 0)), int(seed)])
        dx, dy = rng.uniform(-xy_m, xy_m, 2)
        return {"dx": float(dx), "dy": float(dy), "dyaw": float(rng.uniform(-yaw_rad, yaw_rad))}
    return None


def base_offset_matrix(offset, kwargs):
    """4x4 world transform applying `offset`, pivoting on the nominal base origin; None when `offset` is None.

    Rejects single-arm-pair embodiments: their two roots straddle the pivot, so one rigid move of "the
    base" is ambiguous there.
    """
    if offset is None:
        return None
    if not kwargs["dual_arm_embodied"]:
        raise ValueError("base_pose_offset needs a single dual-arm embodiment (len(embodiment) == 1)")
    pivot = np.zeros(3)
    pivot[:2] = kwargs["left_embodiment_config"]["robot_pose"][0][:2]
    cos, sin = np.cos(offset["dyaw"]), np.sin(offset["dyaw"])
    matrix = np.eye(4)
    matrix[:2, :2] = [[cos, -sin], [sin, cos]]
    matrix[:3, 3] = pivot - matrix[:3, :3] @ pivot + [offset["dx"], offset["dy"], 0.0]
    return matrix


def pose_xy_yaw(matrix):
    """World (x, y, yaw) of a 4x4 pose; yaw is the heading of its x axis."""
    return float(matrix[0, 3]), float(matrix[1, 3]), float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def robot_penetrations(scene, robot_links, tol_m=PENETRATION_TOL_M):
    """Deepest penetration per (robot link, other entity) pair, as [(link_name, other_name, depth_m)].

    Reads the contacts of the scene's last physics step, so call it after setup has stepped the scene.
    The floor is ignored (the wheels rest on it); everything else, scene objects included, counts.
    """
    robot = {link.entity for link in robot_links}
    deepest = {}
    for contact in scene.get_contacts():
        first, second = contact.bodies[0].entity, contact.bodies[1].entity
        if (first in robot) == (second in robot):
            continue
        link, other = (first, second) if first in robot else (second, first)
        if other.name in IGNORED_CONTACT_NAMES:
            continue
        depth = -min(point.separation for point in contact.points)
        if depth > tol_m:
            key = (link.name, other.name)
            deepest[key] = max(depth, deepest.get(key, 0.0))
    return [(*key, depth) for key, depth in sorted(deepest.items(), key=lambda kv: -kv[1])]
