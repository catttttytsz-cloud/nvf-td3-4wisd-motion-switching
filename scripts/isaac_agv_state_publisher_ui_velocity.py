#!/usr/bin/env python3
"""Isaac Sim UI training bridge with only minimal steering velocity-drive changes.

It keeps the original headless bridge ROS initialization order and shell environment.
The USD steering drive is adjusted after stage load and before PhysX initialization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        default=os.environ.get("NVF_ISAAC_USD_PATH", ""),
        help="Path to the training USD scene.",
    )
    parser.add_argument(
        "--publish-rate-hz",
        type=float,
        default=20.0,
        help="State-topic publication rate in simulation-time hertz.",
    )
    parser.add_argument(
        "--max-sim-rtf",
        type=float,
        default=0.0,
        help="Maximum simulation real-time factor; values <= 0 disable limiting.",
    )
    parsed, _unknown = parser.parse_known_args()

    
    parsed.ros_domain_id = 70
    parsed.rmw = "rmw_fastrtps_cpp"
    parsed.articulation_path = "/World/amr/amr"
    parsed.base_prim_path = "/World/amr/amr/base_footprint"
    parsed.base_link_path = "/World/amr/amr/base_footprint/base_link"
    parsed.ensure_ground = True
    parsed.ground_prim_path = "/World/defaultGroundPlane"
    parsed.ground_z = 0.0
    parsed.ground_size = 200.0
    parsed.publish_clock = True
    parsed.disable_usd_graphs = True
    parsed.headless = False
    parsed.status_interval = 100
    parsed.control_backend = "core"
    parsed.planar_twist = False
    parsed.fall_warning_z = -1.0
    return parsed

args = parse_args()

os.environ.setdefault("ROS_DISTRO", "humble")
os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
os.environ["RMW_IMPLEMENTATION"] = args.rmw

# SimulationApp must be created before importing most Isaac/Omniverse modules.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

try:
    from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
except Exception:
    from omni.isaac.core.utils.extensions import enable_extension  # type: ignore # noqa: E402

enable_extension("isaacsim.ros2.bridge")
# Dynamic Control is optional in newer Isaac Sim, but when available it is the
# most direct backend for setting per-DOF targets in this standalone bridge.
for _ext in ("omni.isaac.dynamic_control", "isaacsim.core.api"):
    try:
        enable_extension(_ext)
    except Exception:
        pass
for _ in range(30):
    simulation_app.update()

import numpy as np  # noqa: E402
try:
    import rclpy  # noqa: E402
except ModuleNotFoundError as exc:
    print("[AGV standalone bridge][ERROR] Failed to import rclpy after enabling isaacsim.ros2.bridge.")
    simulation_app.close()
    raise exc

from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from isaacsim.core.utils.transformations import pose_from_tf_matrix  # noqa: E402
from pxr import UsdGeom, UsdPhysics, Gf  # noqa: E402

try:
    from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
except Exception:
    from omni.isaac.core.utils.types import ArticulationAction  # type: ignore # noqa: E402


# Keep the same order used by neural_vector_field.config.
STEERING_JOINT_NAMES = [
    "rear_left_steering_joint", "front_left_steering_joint",
    "rear_right_steering_joint", "front_right_steering_joint",
]
DRIVING_JOINT_NAMES = [
    "rear_left_wheel_joint", "front_left_wheel_joint",
    "rear_right_wheel_joint", "front_right_wheel_joint",
]
JOINT_STATE_NAMES = STEERING_JOINT_NAMES + DRIVING_JOINT_NAMES

# Steering velocity-drive parameters.  The existing wheel drives use stiffness=0
# and high damping; steering is changed to the same velocity-drive form before
# the PhysX articulation is initialized.
STEERING_VELOCITY_DRIVE_STIFFNESS = 0.0
STEERING_VELOCITY_DRIVE_DAMPING = 10000000.0
STEERING_VELOCITY_DRIVE_MAX_FORCE = 1075.0


def configure_steering_velocity_drive(stage) -> None:
    """Configure the four steering joints before World/PhysX initialization."""
    for joint_name in STEERING_JOINT_NAMES:
        joint_prim = stage.GetPrimAtPath(f"/World/amr/amr/joints/{joint_name}")
        if not joint_prim.IsValid():
            raise RuntimeError(f"Missing steering joint prim: {joint_name}")

        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

        drive.GetStiffnessAttr().Set(STEERING_VELOCITY_DRIVE_STIFFNESS)
        drive.GetDampingAttr().Set(STEERING_VELOCITY_DRIVE_DAMPING)
        drive.GetMaxForceAttr().Set(STEERING_VELOCITY_DRIVE_MAX_FORCE)
        drive.GetTargetVelocityAttr().Set(0.0)

        print(
            f"[AGV standalone bridge] steering velocity drive: {joint_name}, "
            f"stiffness={STEERING_VELOCITY_DRIVE_STIFFNESS}, "
            f"damping={STEERING_VELOCITY_DRIVE_DAMPING}, "
            f"max_force={STEERING_VELOCITY_DRIVE_MAX_FORCE}"
        )

WHEEL_DIR_PRIMS: Dict[str, str] = {
    "lb_wheel": "/World/amr/amr/rear_left_steering_link",
    "lf_wheel": "/World/amr/amr/front_left_steering_link",
    "rb_wheel": "/World/amr/amr/rear_right_steering_link",
    "rf_wheel": "/World/amr/amr/front_right_steering_link",
}

GROUND_ROOT_CANDIDATES = (
    "/World/defaultGroundPlane",
    "/World/GroundPlane",
    "/World/groundPlane",
    "/World/ground",
    "/World/Floor",
    "/World/floor",
)


def _is_ground_like_path(path: str) -> bool:
    lowered = path.lower()
    return any(token in lowered for token in ("ground", "floor", "terrain", "plane"))


def find_ground_collision_prims(stage) -> list[str]:
    """Return collision prim paths that look like a static ground/floor."""
    found: list[str] = []
    for prim in list(stage.Traverse()):
        try:
            if not prim.IsValid():
                continue
            path = str(prim.GetPath())
            if not _is_ground_like_path(path):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                found.append(path)
        except Exception:
            continue
    return found


def ensure_fallback_ground(stage, world, prim_path: str, ground_z: float, size: float) -> str:
    """Make sure a collision ground exists. Create a static thin cube fallback if needed."""
    existing = find_ground_collision_prims(stage)
    if existing:
        print("[AGV standalone bridge] Ground collider(s) found:")
        for path in existing[:8]:
            print(f"  ground collider: {path}")
        if len(existing) > 8:
            print(f"  ... {len(existing) - 8} more ground-like collision prim(s)")
        return "existing"

    print("[AGV standalone bridge][WARN] No ground/floor collision prim found; creating fallback static ground.")
    # Prefer Isaac Core's default ground plane when available.  It creates the
    # correct physics/collider setup for the current World.
    try:
        if hasattr(world.scene, "add_default_ground_plane"):
            world.scene.add_default_ground_plane()
            print("[AGV standalone bridge] Added default ground plane through world.scene.add_default_ground_plane().")
            return "world.scene.add_default_ground_plane"
    except Exception as exc:
        print(f"[AGV standalone bridge][WARN] add_default_ground_plane failed, using USD collision cube fallback: {exc}")

    # USD-only fallback: a static cube collider whose top surface is at ground_z.
    # This is deliberately independent of OmniGraph and is sufficient to stop
    # free-fall when the loaded stage has no ground collider.
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    half_thickness = 0.025
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(ground_z - half_thickness)))
    xform.AddScaleOp().Set(Gf.Vec3d(float(size), float(size), float(2.0 * half_thickness)))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    print(f"[AGV standalone bridge] Added fallback collision ground: {prim_path}, top_z={ground_z}, size={size}m")
    return "usd_collision_cube"


def quat_to_world_forward_vector(quat_wxyz):
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return R.from_quat(quat_xyzw).apply([1.0, 0.0, 0.0])


def stamp_from_sim_time(sim_time_sec: float) -> Time:
    sec = int(sim_time_sec)
    nanosec = int((sim_time_sec - sec) * 1.0e9)
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def matrix_from_prim_world(prim):
    return np.asarray(np.transpose(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)), dtype=np.float64)


def pose_from_prim_world(prim):
    return pose_from_tf_matrix(matrix_from_prim_world(prim))


def disable_omnigraph_prims(stage) -> list[str]:
    """Deactivate OmniGraph/ActionGraph prims without touching stale child handles.

    Deactivating a parent prim invalidates the prim handles of its descendants.
    Therefore we first collect path strings, reduce them to top-level graph roots,
    and then reacquire each prim immediately before SetActive(False).
    """
    candidate_paths: list[str] = []

    for prim in list(stage.Traverse()):
        try:
            if not prim.IsValid():
                continue
            path = str(prim.GetPath())
            type_name = str(prim.GetTypeName())
            basename = path.rsplit("/", 1)[-1]
        except Exception:
            # Some USD/OmniGraph edits can invalidate a prim handle while traversing.
            # Ignore it and continue collecting the remaining paths.
            continue

        if (
            type_name == "OmniGraph"
            or basename == "Graph"
            or basename.startswith("ActionGraph")
            or "ActionGraph_" in basename
        ):
            candidate_paths.append(path)

    # If /A/Graph is deactivated, /A/Graph/Graph becomes invalid. Keep only roots.
    selected_paths: list[str] = []
    for path in sorted(candidate_paths, key=lambda p: p.count("/")):
        if any(path == root or path.startswith(root + "/") for root in selected_paths):
            continue
        selected_paths.append(path)

    disabled: list[str] = []
    for path in selected_paths:
        try:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            prim.SetActive(False)
            disabled.append(path)
        except Exception as exc:
            print(f"[AGV standalone bridge][WARN] Failed to deactivate graph prim {path}: {exc}")

    return disabled


class StandaloneArticulationBridge:
    """Read and command the AMR articulation without relying on USD ActionGraphs."""

    def __init__(self, articulation_path: str, preferred_backend: str = "dynamic_control"):
        self.articulation_path = articulation_path
        self.preferred_backend = preferred_backend
        self.joint_names = JOINT_STATE_NAMES
        self.mode = "uninitialized"
        self.articulation = None
        self.all_dof_names: list[str] = []
        self.indices: list[int] | None = None
        self.steering_indices: list[int] | None = None
        self.driving_indices: list[int] | None = None
        self.dc = None
        self.dc_articulation = None
        self.dc_dofs = None
        self.latest_steering_cmd = np.zeros(4, dtype=np.float64)
        self.latest_driving_cmd = np.zeros(4, dtype=np.float64)
        self.steering_cmd_seen = False
        self.driving_cmd_seen = False
        self.last_steering_wall_time: Optional[float] = None
        self.last_driving_wall_time: Optional[float] = None
        self.apply_count = 0
        self.apply_failure_count = 0
        self.last_apply_error = ""
        self.warned_apply_failure = False
        self.warned_read_failure = False

    def initialize(self) -> bool:
        if self.preferred_backend == "dynamic_control":
            candidates = ("dynamic_control", "core")
        elif self.preferred_backend == "core":
            candidates = ("core", "dynamic_control")
        else:
            candidates = ("dynamic_control", "core")

        for backend in candidates:
            if backend == "dynamic_control" and self._initialize_dynamic_control():
                self._print_mapping()
                return True
            if backend == "core" and self._initialize_core_articulation():
                self._print_mapping()
                return True

        print("[AGV standalone bridge][ERROR] No articulation interface is available; cannot train headless.")
        return False

    def _initialize_core_articulation(self) -> bool:
        """Initialize Isaac Core articulation control.

        Prefer SingleArticulation for one robot.  In Isaac Sim 5.1 the public
        documentation shows SingleArticulation.apply_action(ArticulationAction)
        as the standard one-robot path.  The Articulation view class is kept as
        a fallback, but its availability varies across Isaac Sim versions in
        headless runs.
        """
        try:
            articulation = None
            init_kind = ""
            try:
                from isaacsim.core.prims import SingleArticulation  # type: ignore
                articulation = SingleArticulation(
                    prim_path=self.articulation_path,
                    name="amr_single_articulation",
                )
                init_kind = "SingleArticulation"
            except Exception as exc_single:
                print(f"[AGV standalone bridge][WARN] SingleArticulation init path unavailable: {exc_single}")
                try:
                    from isaacsim.core.prims import Articulation  # type: ignore
                except Exception:
                    from omni.isaac.core.articulations import Articulation  # type: ignore
                articulation = Articulation(
                    prim_paths_expr=self.articulation_path,
                    name="amr_articulation_view",
                )
                init_kind = "ArticulationView"

            articulation.initialize()

            names = None
            for attr_name in ("dof_names", "joint_names"):
                if hasattr(articulation, attr_name):
                    value = getattr(articulation, attr_name)
                    if value is not None:
                        names = list(value)
                        break
            if names is None and hasattr(articulation, "get_dof_names"):
                names = list(articulation.get_dof_names())
            if names is None and hasattr(articulation, "get_joint_names"):
                names = list(articulation.get_joint_names())
            if names is None:
                print("[AGV standalone bridge][WARN] Core articulation did not expose DOF names.")
                return False

            indices: list[int] = []
            missing = []
            for joint_name in self.joint_names:
                if joint_name in names:
                    indices.append(names.index(joint_name))
                elif hasattr(articulation, "get_dof_index"):
                    try:
                        indices.append(int(articulation.get_dof_index(joint_name)))
                    except Exception:
                        missing.append(joint_name)
                else:
                    missing.append(joint_name)
            if missing:
                print(f"[AGV standalone bridge][WARN] Core articulation missing joints: {missing}")
                return False

            self.articulation = articulation
            self.all_dof_names = list(names)
            self.indices = indices
            self.steering_indices = indices[:4]
            self.driving_indices = indices[4:]
            self.mode = "core_single_articulation" if init_kind == "SingleArticulation" else "core_articulation_view"
            return True
        except Exception as exc:
            print(f"[AGV standalone bridge][WARN] Core articulation init failed: {exc}")
            return False

    def _initialize_dynamic_control(self) -> bool:
        try:
            from omni.isaac.dynamic_control import _dynamic_control  # type: ignore
            dc = _dynamic_control.acquire_dynamic_control_interface()
            articulation = dc.get_articulation(self.articulation_path)
            if not articulation:
                print(f"[AGV standalone bridge][WARN] Dynamic-control articulation not found: {self.articulation_path}")
                return False

            dofs = {}
            missing = []
            for joint_name in self.joint_names:
                dof = dc.find_articulation_dof(articulation, joint_name)
                if not dof:
                    missing.append(joint_name)
                else:
                    dofs[joint_name] = dof
            if missing:
                print(f"[AGV standalone bridge][WARN] Dynamic-control missing joints: {missing}")
                return False

            self.dc = dc
            self.dc_articulation = articulation
            self.dc_dofs = dofs
            self.all_dof_names = [str(name) for name in self.joint_names]
            self.indices = list(range(len(self.joint_names)))
            self.steering_indices = list(range(4))
            self.driving_indices = list(range(4, 8))
            self.mode = "dynamic_control"
            return True
        except Exception as exc:
            print(f"[AGV standalone bridge][WARN] Dynamic-control init failed: {exc}")
            return False

    def _print_mapping(self) -> None:
        print(f"[AGV standalone bridge] articulation_path: {self.articulation_path}")
        print(f"[AGV standalone bridge] preferred_backend: {self.preferred_backend}")
        print(f"[AGV standalone bridge] articulation_mode: {self.mode}")
        if self.all_dof_names:
            print("[AGV standalone bridge] all DOF names:")
            for idx, name in enumerate(self.all_dof_names):
                print(f"  [{idx:02d}] {name}")
        print("[AGV standalone bridge] training joint mapping:")
        if self.indices is not None:
            for joint_name, idx in zip(self.joint_names, self.indices):
                print(f"  {joint_name} -> {idx}")

    def on_steering_command(self, msg: JointState) -> None:
        values = self._extract_values(msg, self.steering_cmd_seen, STEERING_JOINT_NAMES, "velocity")
        if values is not None:
            self.latest_steering_cmd = values
            self.steering_cmd_seen = True
            self.last_steering_wall_time = time.perf_counter()

    def on_driving_command(self, msg: JointState) -> None:
        values = self._extract_values(msg, self.driving_cmd_seen, DRIVING_JOINT_NAMES, "velocity")
        if values is not None:
            self.latest_driving_cmd = values
            self.driving_cmd_seen = True
            self.last_driving_wall_time = time.perf_counter()

    @staticmethod
    def _extract_values(msg: JointState, _seen: bool, expected_names: Sequence[str], field: str) -> Optional[np.ndarray]:
        raw = getattr(msg, field)
        if len(raw) < len(expected_names):
            return None
        if msg.name:
            value_dict = dict(zip(msg.name, raw))
            if all(name in value_dict for name in expected_names):
                return np.asarray([value_dict[name] for name in expected_names], dtype=np.float64)
        # The training node publishes in the expected order, so positional fallback is valid.
        return np.asarray(list(raw[:len(expected_names)]), dtype=np.float64)

    def apply_latest_command(self) -> None:
        # Do not repeatedly send zero commands and count failures before the
        # training node (or self-test) has actually sent commands.
        if not (self.steering_cmd_seen or self.driving_cmd_seen):
            return
        if self.mode.startswith("core_") and self.articulation is not None:
            self._apply_core_action()
        elif self.mode == "dynamic_control" and self.dc is not None and self.dc_dofs is not None:
            self._apply_dynamic_control()

    def _record_apply_failure(self, backend: str, exc: Exception) -> None:
        self.apply_failure_count += 1
        self.last_apply_error = f"{backend}: {exc}"
        if not self.warned_apply_failure:
            print(f"[AGV standalone bridge][WARN] {backend} command application failed: {exc}")
            self.warned_apply_failure = True

    def _apply_core_action(self) -> None:
        try:
            # Use list inputs to avoid version-specific numpy shape handling.
            # Attach joint_names defensively for older/transition APIs that look
            # for this attribute even though the documented 5.1 ArticulationAction
            # constructor uses joint_indices.
            if self.steering_indices is not None and self.steering_cmd_seen:
                steering_action = ArticulationAction(
                    joint_velocities=[float(v) for v in self.latest_steering_cmd],
                    joint_indices=[int(i) for i in self.steering_indices],
                )
                try:
                    steering_action.joint_names = list(STEERING_JOINT_NAMES)
                except Exception:
                    pass
                self.articulation.apply_action(steering_action)
            if self.driving_indices is not None and self.driving_cmd_seen:
                driving_action = ArticulationAction(
                    joint_velocities=[float(v) for v in self.latest_driving_cmd],
                    joint_indices=[int(i) for i in self.driving_indices],
                )
                try:
                    driving_action.joint_names = list(DRIVING_JOINT_NAMES)
                except Exception:
                    pass
                self.articulation.apply_action(driving_action)
            self.apply_count += 1
            self.last_apply_error = ""
        except Exception as exc:
            self._record_apply_failure("core", exc)

    def _apply_dynamic_control(self) -> None:
        try:
            self.dc.wake_up_articulation(self.dc_articulation)
            for name, target in zip(STEERING_JOINT_NAMES, self.latest_steering_cmd):
                self.dc.set_dof_velocity_target(self.dc_dofs[name], float(target))
            for name, target in zip(DRIVING_JOINT_NAMES, self.latest_driving_cmd):
                self.dc.set_dof_velocity_target(self.dc_dofs[name], float(target))
            self.apply_count += 1
            self.last_apply_error = ""
        except Exception as exc:
            self._record_apply_failure("dynamic_control", exc)

    def read_joint_state(self):
        if self.mode.startswith("core_") and self.articulation is not None and self.indices is not None:
            try:
                positions = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
                velocities = np.asarray(self.articulation.get_joint_velocities(), dtype=np.float64).reshape(-1)
                return [float(positions[i]) for i in self.indices], [float(velocities[i]) for i in self.indices]
            except Exception as exc:
                if not self.warned_read_failure:
                    print(f"[AGV standalone bridge][WARN] Core read joint state failed: {exc}")
                    self.warned_read_failure = True
                return None
        if self.mode == "dynamic_control" and self.dc is not None and self.dc_dofs is not None:
            try:
                positions = [float(self.dc.get_dof_position(self.dc_dofs[name])) for name in self.joint_names]
                velocities = [float(self.dc.get_dof_velocity(self.dc_dofs[name])) for name in self.joint_names]
                return positions, velocities
            except Exception as exc:
                if not self.warned_read_failure:
                    print(f"[AGV standalone bridge][WARN] Dynamic-control read joint state failed: {exc}")
                    self.warned_read_failure = True
                return None
        return None

    def command_status(self) -> str:
        now = time.perf_counter()
        steer_age = "--" if self.last_steering_wall_time is None else f"{now - self.last_steering_wall_time:.3f}s"
        drive_age = "--" if self.last_driving_wall_time is None else f"{now - self.last_driving_wall_time:.3f}s"
        drive_mean = float(np.mean(np.abs(self.latest_driving_cmd)))
        steer_mean = float(np.mean(np.abs(self.latest_steering_cmd)))
        err = "" if not self.last_apply_error else f" err={self.last_apply_error}"
        return (
            f"backend={self.mode} "
            f"cmd_seen(S/D)={int(self.steering_cmd_seen)}/{int(self.driving_cmd_seen)} "
            f"age(S/D)={steer_age}/{drive_age} "
            f"|steer|mean={steer_mean:.3f} |wheel|mean={drive_mean:.3f} "
            f"apply={self.apply_count} fail={self.apply_failure_count}{err}"
        )


class BaseVelocityReader:
    def __init__(self, base_paths: Sequence[str]):
        self.base_paths = list(base_paths)
        self.mode = "uninitialized"
        self.rigid = None
        self.rigid_path = None
        self.warned = False

    def initialize(self) -> None:
        try:
            from isaacsim.core.prims import RigidPrim  # type: ignore
        except Exception:
            try:
                from omni.isaac.core.prims import RigidPrim  # type: ignore
            except Exception as exc:
                print(f"[AGV standalone bridge][WARN] RigidPrim import failed; base twist will be zero until fixed: {exc}")
                return
        for path in self.base_paths:
            try:
                rigid = RigidPrim(prim_paths_expr=path, name=f"base_velocity_reader_{abs(hash(path))}")
                rigid.initialize()
                self.rigid = rigid
                self.rigid_path = path
                self.mode = "rigid_prim"
                print(f"[AGV standalone bridge] Base velocity reader initialized via RigidPrim: {path}")
                return
            except Exception as exc:
                print(f"[AGV standalone bridge][WARN] RigidPrim velocity reader failed for {path}: {exc}")

    def read_world_velocity(self):
        if self.mode != "rigid_prim" or self.rigid is None:
            return None
        try:
            lin = np.asarray(self.rigid.get_linear_velocities()[0], dtype=np.float64)
            ang = np.asarray(self.rigid.get_angular_velocities()[0], dtype=np.float64)
            return lin, ang
        except Exception as exc:
            if not self.warned:
                print(f"[AGV standalone bridge][WARN] RigidPrim velocity read failed; base twist will be zero until fixed: {exc}")
                self.warned = True
            return None


if not args.usd or not os.path.exists(args.usd):
    print(f"[AGV standalone bridge][ERROR] USD file not found: {args.usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[AGV standalone bridge] Loading USD: {args.usd}")
omni.usd.get_context().open_stage(args.usd)
for _ in range(20):
    simulation_app.update()

stage = get_current_stage()
if args.disable_usd_graphs:
    disabled_graphs = disable_omnigraph_prims(stage)
    print(f"[AGV standalone bridge] Deactivated {len(disabled_graphs)} USD graph prim(s) for this run.")
    for path in disabled_graphs:
        print(f"  graph disabled: {path}")

# Must execute before world.initialize_physics() and world.reset().
configure_steering_velocity_drive(stage)

if not rclpy.ok():
    rclpy.init(args=None)
ros_node = rclpy.create_node("isaac_agv_standalone_bridge")
clock_pub = ros_node.create_publisher(Clock, "/clock", 10) if args.publish_clock else None
joint_pub = ros_node.create_publisher(JointState, "/joint_states", 10)
pose_pub = ros_node.create_publisher(PoseStamped, "/agv/base_link_pose", 10)
vel_pub = ros_node.create_publisher(TwistStamped, "/agv/base_link_twist", 10)
data_pub = ros_node.create_publisher(Float32MultiArray, "/agv/wheel_vector_data", 10)
tf_pub = ros_node.create_publisher(TFMessage, "/tf", 10)
tf_static_qos = QoSProfile(depth=1)
tf_static_qos.reliability = ReliabilityPolicy.RELIABLE
tf_static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
tf_static_pub = ros_node.create_publisher(TFMessage, "/tf_static", tf_static_qos)

world = World.instance() if World.instance() is not None else World()
if hasattr(world, "initialize_physics"):
    world.initialize_physics()
stage = get_current_stage()
if args.ensure_ground:
    ensure_fallback_ground(stage, world, args.ground_prim_path, args.ground_z, args.ground_size)
# Reset after the fallback ground is present, so PhysX initializes the collider before simulation starts.
world.reset()
stage = get_current_stage()
base_prim = stage.GetPrimAtPath(args.base_prim_path)
base_link_prim = stage.GetPrimAtPath(args.base_link_path)
if not base_prim.IsValid():
    print(f"[AGV standalone bridge][ERROR] Missing base prim: {args.base_prim_path}")
if not base_link_prim.IsValid():
    print(f"[AGV standalone bridge][WARN] Missing base_link prim: {args.base_link_path}; static base TF will use identity.")

art_bridge = StandaloneArticulationBridge(args.articulation_path, preferred_backend=args.control_backend)
if not art_bridge.initialize():
    simulation_app.close()
    sys.exit(2)
vel_reader = BaseVelocityReader([args.base_prim_path, args.base_link_path])
vel_reader.initialize()

ros_node.create_subscription(JointState, "/steering_joints_controller/command", art_bridge.on_steering_command, 10)
ros_node.create_subscription(JointState, "/driving_joints_controller/command", art_bridge.on_driving_command, 10)

publish_period = 1.0 / max(args.publish_rate_hz, 1.0e-6)
state = {
    "sim_time": 0.0,
    "next_publish_time": 0.0,
    "publish_count": 0,
    "wall_start": time.perf_counter(),
    "last_pose_time": None,
    "last_translation": None,
    "last_rotation": None,
    "published_static_tf": False,
    "velocity_source": "uninitialized",
    "fall_warning_printed": False,
}


def publish_clock(stamp: Time) -> None:
    if clock_pub is None:
        return
    clock_msg = Clock()
    clock_msg.clock = stamp
    clock_pub.publish(clock_msg)


def publish_joint_state(stamp: Time) -> None:
    read_result = art_bridge.read_joint_state()
    if read_result is None:
        return
    positions, velocities = read_result
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = JOINT_STATE_NAMES
    msg.position = positions
    msg.velocity = velocities
    msg.effort = []
    joint_pub.publish(msg)


def publish_static_base_tf(stamp: Time) -> None:
    if state["published_static_tf"]:
        return
    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = "base_footprint"
    tf_msg.child_frame_id = "base_link"

    stage = get_current_stage()
    parent = stage.GetPrimAtPath(args.base_prim_path)
    child = stage.GetPrimAtPath(args.base_link_path)
    if parent.IsValid() and child.IsValid():
        try:
            parent_world = matrix_from_prim_world(parent)
            child_world = matrix_from_prim_world(child)
            local = np.linalg.inv(parent_world) @ child_world
            trans, quat = pose_from_tf_matrix(local)
            tf_msg.transform.translation.x = float(trans[0])
            tf_msg.transform.translation.y = float(trans[1])
            tf_msg.transform.translation.z = float(trans[2])
            tf_msg.transform.rotation.w = float(quat[0])
            tf_msg.transform.rotation.x = float(quat[1])
            tf_msg.transform.rotation.y = float(quat[2])
            tf_msg.transform.rotation.z = float(quat[3])
        except Exception as exc:
            print(f"[AGV standalone bridge][WARN] Failed to compute base_footprint->base_link TF from USD; using identity: {exc}")
            tf_msg.transform.rotation.w = 1.0
    else:
        tf_msg.transform.rotation.w = 1.0

    tf_static_pub.publish(TFMessage(transforms=[tf_msg]))
    state["published_static_tf"] = True


def publish_world_base_tf(stamp: Time, translation, quaternion) -> None:
    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = "world"
    tf_msg.child_frame_id = "base_footprint"
    tf_msg.transform.translation.x = float(translation[0])
    tf_msg.transform.translation.y = float(translation[1])
    tf_msg.transform.translation.z = float(translation[2])
    tf_msg.transform.rotation.w = float(quaternion[0])
    tf_msg.transform.rotation.x = float(quaternion[1])
    tf_msg.transform.rotation.y = float(quaternion[2])
    tf_msg.transform.rotation.z = float(quaternion[3])
    tf_pub.publish(TFMessage(transforms=[tf_msg]))


def compute_base_twist(sim_time: float, translation, quaternion):
    """Compute /agv/base_link_twist with the same frame convention as the old GUI publisher.

    The old Script-Editor publisher read the physical linear/angular velocity of
    BASE_PRIM_PATH and rotated both vectors into the base local frame before
    publishing them.  We keep that convention here: header.frame_id remains
    "world" for compatibility with the existing training/visualization nodes,
    while twist components are the base-frame velocity components used by the
    original training code.
    """
    translation = np.asarray(translation, dtype=np.float64)
    rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])

    # Default and paper-training-compatible path: exactly follow the early GUI
    # Script-Editor publisher convention.
    #   1) read world-frame linear/angular velocity from RigidPrim(BASE_PRIM_PATH),
    #   2) rotate both vectors into the base-local frame,
    #   3) publish the full local 3D twist.
    physical = vel_reader.read_world_velocity()
    if physical is not None:
        lin_world, ang_world = physical
        lin_local = rotation.inv().apply(np.asarray(lin_world, dtype=np.float64))
        ang_local = rotation.inv().apply(np.asarray(ang_world, dtype=np.float64))
        state["velocity_source"] = f"legacy_rigid:{vel_reader.rigid_path}"
        return lin_local, ang_local

    state["velocity_source"] = "legacy_rigid_unavailable"
    return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

def publish_state_sample(stamp: Time) -> None:
    stage = get_current_stage()
    publish_joint_state(stamp)
    publish_static_base_tf(stamp)

    base_prim = stage.GetPrimAtPath(args.base_prim_path)
    if base_prim.IsValid():
        translation, quaternion = pose_from_prim_world(base_prim)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])
        pose_msg.pose.orientation.w = float(quaternion[0])
        pose_msg.pose.orientation.x = float(quaternion[1])
        pose_msg.pose.orientation.y = float(quaternion[2])
        pose_msg.pose.orientation.z = float(quaternion[3])
        pose_pub.publish(pose_msg)
        publish_world_base_tf(stamp, translation, quaternion)

        lin_vel_local, ang_vel_local = compute_base_twist(state["sim_time"], translation, quaternion)
        if (not state["fall_warning_printed"]) and float(translation[2]) < float(args.fall_warning_z):
            print(
                f"[AGV standalone bridge][WARN] base z={float(translation[2]):.3f} below {args.fall_warning_z:.3f}. "
                "The robot is probably falling; check ground collider and initial pose.",
                flush=True,
            )
            state["fall_warning_printed"] = True
        twist_msg = TwistStamped()
        twist_msg.header = pose_msg.header
        twist_msg.twist.linear.x = float(lin_vel_local[0])
        twist_msg.twist.linear.y = float(lin_vel_local[1])
        twist_msg.twist.linear.z = float(lin_vel_local[2])
        twist_msg.twist.angular.x = float(ang_vel_local[0])
        twist_msg.twist.angular.y = float(ang_vel_local[1])
        twist_msg.twist.angular.z = float(ang_vel_local[2])
        vel_pub.publish(twist_msg)

    flat = []
    for name, link_path in WHEEL_DIR_PRIMS.items():
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            print(f"[AGV standalone bridge][WARN] Missing wheel dir prim: {name} -> {link_path}")
            continue
        pos, quat = pose_from_prim_world(link_prim)
        forward = quat_to_world_forward_vector(quat)
        flat.extend([
            float(pos[0]), float(pos[1]), float(pos[2]),
            0.0, 0.0, 0.0,
            float(forward[0]), float(forward[1]), float(forward[2]),
        ])

    vector_msg = Float32MultiArray()
    vector_msg.data = flat
    data_pub.publish(vector_msg)


def apply_max_sim_rtf_limit() -> None:
    """Limit sim_time / wall_time without changing physics dt or publish frequency."""
    max_rtf = float(args.max_sim_rtf)
    if max_rtf <= 0.0:
        return
    sim_time = float(state.get("sim_time", 0.0))
    if sim_time <= 0.0:
        return
    wall_elapsed = time.perf_counter() - float(state["wall_start"])
    target_wall_elapsed = sim_time / max(max_rtf, 1.0e-9)
    sleep_time = target_wall_elapsed - wall_elapsed
    if sleep_time > 0.0:
        time.sleep(min(sleep_time, 0.02))


def physics_callback(step_size: float):
    state["sim_time"] += float(step_size)
    # Pull command messages even when no status sample is published, then apply
    # the newest command every physics step.
    rclpy.spin_once(ros_node, timeout_sec=0.0)
    art_bridge.apply_latest_command()

    if state["sim_time"] + 1.0e-12 < state["next_publish_time"]:
        return

    state["next_publish_time"] += publish_period
    if state["next_publish_time"] < state["sim_time"]:
        state["next_publish_time"] = state["sim_time"] + publish_period

    state["publish_count"] += 1
    stamp = stamp_from_sim_time(state["sim_time"])

    publish_state_sample(stamp)

    if args.status_interval > 0 and state["publish_count"] % args.status_interval == 0:
        wall_elapsed = max(time.perf_counter() - state["wall_start"], 1.0e-9)
        rtf = state["sim_time"] / wall_elapsed
        read_result = art_bridge.read_joint_state()
        if read_result is None:
            wheel_mean = 0.0
            steer_mean = 0.0
        else:
            pos, vel = read_result
            steer_mean = float(np.mean(np.abs(pos[:4])))
            wheel_mean = float(np.mean(np.abs(vel[4:])))
        print(
            f"[AGV standalone bridge] sim_time={state['sim_time']:.2f}s "
            f"publishes={state['publish_count']} rtf~={rtf:.2f} "
            f"max_rtf={'off' if args.max_sim_rtf <= 0.0 else f'{args.max_sim_rtf:.2f}'} "
            f"rate={args.publish_rate_hz:.1f}Hz(sim) "
            f"joint(|steer_pos|mean={steer_mean:.3e}, |wheel_vel|mean={wheel_mean:.3e}) "
            f"vel_source={state['velocity_source']} "
            f"base_z={float(state['last_translation'][2]) if state['last_translation'] is not None else 0.0:.3f} "
            f"{art_bridge.command_status()}",
            flush=True,
        )

    # Publish /clock last for this simulation sample to reduce RViz future
    # extrapolation during high-RTF execution.
    publish_clock(stamp)


world.add_physics_callback("agv_standalone_bridge", callback_fn=physics_callback)
print(
    "[AGV standalone bridge] Started (UI steering-velocity mode). "
    f"usd={args.usd}, "
    f"publish_rate={args.publish_rate_hz:.1f} Hz sim-time, "
    f"max_sim_rtf={'off' if args.max_sim_rtf <= 0.0 else f'{args.max_sim_rtf:.2f}'}, "
    f"control_backend={args.control_backend}, "
    f"velocity=legacy_rigid:{args.base_prim_path}, "
    f"articulation={args.articulation_path}"
)

timeline = omni.timeline.get_timeline_interface()
timeline.play()

try:
    while simulation_app.is_running():
        world.step(render=not args.headless)
        apply_max_sim_rtf_limit()
finally:
    try:
        timeline.stop()
    except Exception:
        pass
    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
