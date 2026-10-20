"""Read-only, per-command MuJoCo diagnostics; angles are explicitly labelled."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np


class SimDiagnostics:
    def __init__(self, path, env, names, addresses, config, context=None, effective_limits=None):
        self.env = env
        self.names = list(names)
        self.addresses = np.asarray(addresses)
        self.joints = env.model.actuator_trnid[:, 0].astype(int)
        self.rom = np.array([config.joint_roms_dict[n] for n in names], dtype=float)
        self.core_rom = self.rom.copy()
        if effective_limits is not None:
            self.rom = np.rad2deg([effective_limits[n] for n in names])
        self.scratch = mujoco.MjData(env.model)
        self.frame = 0
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', buffering=1)
        m, d = env.model, env.data
        self.write(dict(event='start', schema_version=1, context=context,
                        scene_path=str(env.scene_path), joint_names=self.names,
                        frame_skip=env.frame_skip, timestep_s=float(m.opt.timestep),
                        initial_qpos=d.qpos.tolist(), initial_qvel=d.qvel.tolist(),
                        retarget_rom_deg=self.rom.tolist(), core_rom_deg=self.core_rom.tolist(),
                        control_range_rad=m.actuator_ctrlrange.tolist(),
                        control_limited=m.actuator_ctrllimited.tolist(),
                        joint_range_rad=m.jnt_range[self.joints].tolist(),
                        joint_limited=m.jnt_limited[self.joints].tolist(),
                        exclude_signature=m.exclude_signature.tolist(),
                        compiled_model_xml=mujoco_model_xml(m)))

    def write(self, row):
        self.file.write(json.dumps(row, allow_nan=False) + '\n')

    def record(self, target_rad):
        m, d = self.env.model, self.env.data
        # Refresh contacts at the final state on a separate MjData. Never change
        # live solver/warm-start state for logging. Contacts are endpoint samples.
        mujoco.mj_copyData(self.scratch, m, d)
        mujoco.mj_forward(m, self.scratch)
        contacts = []
        for c in self.scratch.contact:
            g1, g2 = int(c.geom1), int(c.geom2)
            contacts.append(dict(
                geom_ids=[g1, g2],
                geom_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (g1, g2)],
                bodies=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in (g1, g2)],
                distance_m=float(c.dist), constraint_active=bool(c.efc_address >= 0)))
        target_rad = np.asarray(target_rad)
        actual = d.qpos[self.addresses]
        target_deg = np.rad2deg(target_rad)
        tol = np.deg2rad(1.0)
        joint_margin = np.minimum(actual - m.jnt_range[self.joints, 0], m.jnt_range[self.joints, 1] - actual)
        ctrl_margin = np.minimum(d.ctrl - m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1] - d.ctrl)
        rom_margin = np.minimum(target_deg - self.rom[:, 0], self.rom[:, 1] - target_deg)
        select = lambda mask: [n for n, flag in zip(self.names, mask) if flag]
        self.write(dict(event='frame', frame=self.frame, wall_time_s=time.time(), sim_time_s=float(d.time),
                        target_rad=target_rad.tolist(), applied_ctrl_rad=d.ctrl.tolist(),
                        actual_deg=np.rad2deg(actual).tolist(), target_deg=target_deg.tolist(),
                        tracking_error_deg=np.rad2deg(d.ctrl - actual).tolist(),
                        qpos=d.qpos.tolist(), qvel=d.qvel.tolist(),
                        near_retarget_limit=select(rom_margin <= 1.0),
                        near_joint_limit=select((joint_margin <= tol) & m.jnt_limited[self.joints].astype(bool)),
                        near_control_limit=select((ctrl_margin <= tol) & m.actuator_ctrllimited.astype(bool)),
                        clipped_joints=select(np.abs(target_rad - d.ctrl) > 1e-6), contacts=contacts))
        self.frame += 1

    def close(self):
        self.file.close()


def mujoco_model_xml(model):
    # Retain the compiled model definition for comparison without touching assets.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'model.xml')
        mujoco.mj_saveLastXML(path, model)
        return Path(path).read_text()


def simulation_joint_limits(model, names):
    """Intersect hinge joint and position actuator bounds in MuJoCo qpos units."""
    result = {}
    for i, name in enumerate(names):
        j = int(model.actuator_trnid[i, 0])
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not np.allclose(model.actuator_gear[i], [1, 0, 0, 0, 0, 0]):
            raise ValueError("Simulation bounds require direct unit-gear hinge actuators")
        low, high = -np.inf, np.inf
        if model.jnt_limited[j]:
            low, high = model.jnt_range[j]
        # The current env clips controls to ctrlrange even if ctrllimited is false.
        low = max(low, model.actuator_ctrlrange[i, 0])
        high = min(high, model.actuator_ctrlrange[i, 1])
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Invalid simulation bounds for {name}: {low}, {high}")
        result[name] = (float(low), float(high))
    return result
