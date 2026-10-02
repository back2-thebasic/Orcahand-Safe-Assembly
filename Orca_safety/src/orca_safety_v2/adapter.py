"""MuJoCo embodiment adapter, independent of retargeter and physical hardware."""
from pathlib import Path
import numpy as np
from .collision import FCLDistanceModel
from .config import load_config


def build_sim_model(env, config_path=None, urdf_path=None):
    import mujoco
    if env.version != 'v1':
        raise ValueError('Collision safety v2 currently supports robot v1 only')
    model = env.model
    joints = model.actuator_trnid[:, 0].astype(int)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
    if not all(n and n.startswith('right_') for n in names):
        raise ValueError('Expected v1 right-hand joint names')
    if len(set(names)) != len(names):
        raise ValueError('Actuator mapping must be one-to-one')
    addresses = model.jnt_qposadr[joints].copy()
    workspace = Path(__file__).resolve().parents[3]
    urdf = Path(urdf_path) if urdf_path else workspace/'orcahand_description/v1/models/urdf/orcahand_right.urdf'
    config, pairs = load_config(config_path)
    collision = FCLDistanceModel(urdf, [workspace], names, model.qpos0[addresses], pairs)
    low = np.maximum(model.jnt_range[joints,0], model.actuator_ctrlrange[:,0])
    high = np.minimum(model.jnt_range[joints,1], model.actuator_ctrlrange[:,1])
    dt = model.opt.timestep * env.frame_skip
    max_step = collision.velocity_limits * dt
    if config.max_step_rad is not None:
        max_step = np.minimum(max_step, config.max_step_rad)
    return collision, config, low, high, max_step, addresses
