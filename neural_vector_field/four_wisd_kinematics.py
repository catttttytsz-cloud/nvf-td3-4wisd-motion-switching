"""Inverse kinematics used to obtain target 8-D actuator commands.

All outputs use the package-wide wheel order:
rear-left, front-left, rear-right, front-right.
"""

from __future__ import annotations

import math
import numpy as np

from neural_vector_field.config import WHEEL_RADIUS


def four_wisd_inverse_kinematics(
    vx: float,
    vy: float,
    wz: float,
    *,
    wheel_radius: float = WHEEL_RADIUS,
    wheel_base: float = 2.03,
    track: float = 1.02,
    wheel_steering_y_offset: float = 0.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Return target actuator command ``[steer(4), drive(4)]``.

    The returned order is ``[RL, FL, RR, FR, v_RL, v_FL, v_RR, v_FR]``.
    This preserves the effective order used by the submitted training/testing
    scripts.
    """
    steering_track = track - 2.0 * wheel_steering_y_offset

    fl_st = fr_st = rl_st = rr_st = 0.0
    v_lf = v_rf = v_lr = v_rr = 0.0

    if abs(wz) > eps:
        if abs(vx) > eps:
            vel_steering_offset = (wz * wheel_steering_y_offset) / wheel_radius

            if abs(vx) > abs(wz * 1.550) and abs(vy) < abs(wz):
                fl_st = math.atan(wz * wheel_base / (2.0 * vx - wz * steering_track))
                fr_st = math.atan(wz * wheel_base / (2.0 * vx + wz * steering_track))
                rl_st = -fl_st
                rr_st = -fr_st

                sgn = math.copysign(1.0, vx)
                hyp_l = math.hypot(vx - wz * steering_track / 2.0, wheel_base * wz / 2.0) / wheel_radius
                hyp_r = math.hypot(vx + wz * steering_track / 2.0, wheel_base * wz / 2.0) / wheel_radius
                v_lf = sgn * hyp_l - vel_steering_offset
                v_rf = -sgn * hyp_r + vel_steering_offset
                v_lr = sgn * hyp_l - vel_steering_offset
                v_rr = -sgn * hyp_r + vel_steering_offset

            elif abs(vx) >= abs(vy) and abs(vy) > abs(wz):
                angle = math.atan(vy / vx)
                fl_st = fr_st = rl_st = rr_st = angle

                sgn = math.copysign(1.0, vx)
                speed = math.hypot(vx, vy) / wheel_radius
                v_lf = sgn * speed
                v_rf = -sgn * speed
                v_lr = sgn * speed
                v_rr = -sgn * speed

            else:
                calc_wz = math.copysign(vx / 1.550, wz)
                fl_st = math.atan(calc_wz * wheel_base / (2.0 * vx - calc_wz * steering_track))
                fr_st = math.atan(calc_wz * wheel_base / (2.0 * vx + calc_wz * steering_track))
                rl_st = -fl_st
                rr_st = -fr_st

                sgn = math.copysign(1.0, vx)
                hyp_l = math.hypot(vx - calc_wz * steering_track / 2.0, wheel_base * calc_wz / 2.0) / wheel_radius
                hyp_r = math.hypot(vx + calc_wz * steering_track / 2.0, wheel_base * calc_wz / 2.0) / wheel_radius
                v_lf = sgn * hyp_l - vel_steering_offset
                v_rf = -sgn * hyp_r + vel_steering_offset
                v_lr = sgn * hyp_l - vel_steering_offset
                v_rr = -sgn * hyp_r + vel_steering_offset

        else:
            fl_st = -math.atan(wheel_base / steering_track)
            fr_st = math.atan(wheel_base / steering_track)
            rl_st = fr_st
            rr_st = fl_st

            factor = math.hypot(steering_track / 2.0, wheel_base / 2.0) / wheel_radius
            v_lf = v_rf = v_lr = v_rr = -wz * factor

    elif (abs(vx) > eps or abs(vy) > eps) and abs(wz) < eps:
        if abs(vx) >= abs(vy):
            angle = math.atan(vy / vx)
            fl_st = fr_st = rl_st = rr_st = angle

            sgn = math.copysign(1.0, vx)
            speed = math.hypot(vx, vy) / wheel_radius
            v_lf = sgn * speed
            v_rf = -sgn * speed
            v_lr = sgn * speed
            v_rr = -sgn * speed
        else:
            fl_st = -math.pi / 2.0
            fr_st = math.pi / 2.0
            rl_st = fr_st
            rr_st = fl_st

            v_lf = -vy / wheel_radius
            v_rf = -vy / wheel_radius
            v_lr = vy / wheel_radius
            v_rr = vy / wheel_radius

    return np.array([rl_st, fl_st, rr_st, fr_st, v_lr, v_lf, -v_rr, -v_rf])

# Backward-compatible alias.
agv_inverse_kinematics = four_wisd_inverse_kinematics
