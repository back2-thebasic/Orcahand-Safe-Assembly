import json
import numpy as np
import pytest
from orca_sim import OrcaHandRight
from orca_safety_v2.teleop import SimulationCollisionSafety


def test_simulation_logging_and_bad_command(tmp_path):
    env=OrcaHandRight(version='v1');env.reset()
    runner=SimulationCollisionSafety(env,log_path=tmp_path/'test.jsonl')
    q=env.data.qpos[runner.addresses].copy()
    info=runner.dispatch(q)
    assert info['solver_status']=='solved'
    info=runner.dispatch(np.full(q.shape,np.nan))
    assert info['fallback']
    assert np.all(np.isfinite(env.data.ctrl))
    runner.close();env.close()
    rows=[json.loads(x) for x in (tmp_path/'test.jsonl').read_text().splitlines()]
    assert len(rows)==3 and rows[-1]['q_nominal']==[None]*17
    assert rows[0]['robot']=='v1_right'


def test_monitor_does_not_change_commands(tmp_path):
    env=OrcaHandRight(version='v1');env.reset()
    runner=SimulationCollisionSafety(env,log_path=tmp_path/'monitor.jsonl',monitor_only=True)
    q=env.data.qpos[runner.addresses].copy();q[6]+=.03
    runner.dispatch(q)
    np.testing.assert_allclose(env.data.ctrl,q,atol=1e-7)
    runner.close();env.close()


def test_sink_modes_and_dispatch(tmp_path):
    from orca_teleop.sim import OrcaHandSimSink
    from orca_core import OrcaJointPositions
    with pytest.raises(ValueError):OrcaHandSimSink(collision_safety=True,version='v2')
    with pytest.raises(ValueError):OrcaHandSimSink(collision_safety=True,safety=True,version='v1')
    env=OrcaHandRight(version='v1');env.reset()
    runner=SimulationCollisionSafety(env,log_path=tmp_path/'sink.jsonl')
    sink=OrcaHandSimSink(version='v1',collision_safety=True)
    sink._env=env;sink._collision_executor=runner
    sink._actuator_joint_names=[n.removeprefix('right_') for n in runner.model.joint_names]
    q=env.data.qpos[runner.addresses].copy()
    action=OrcaJointPositions(dict(zip(sink._actuator_joint_names,np.rad2deg(q))))
    sink.dispatch_action(action)
    assert runner.frame==1
    sink.dispatch_action(OrcaJointPositions({}))
    assert runner.filter.failure_count==1
    runner.close();env.close()


def test_sink_waits_at_measured_reset_pose(tmp_path,monkeypatch):
    import mujoco
    from orca_teleop.sim import OrcaHandSimSink
    class NoRenderer:
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
    monkeypatch.setattr(mujoco,'Renderer',NoRenderer)
    sink=OrcaHandSimSink(version='v1',render_mode='rgb_array',collision_safety=True,
                         collision_safety_log=str(tmp_path/'startup.jsonl'))
    try:
        sink.connect()
        current=sink._env.data.qpos[sink._joint_qpos_adr].copy()
        np.testing.assert_allclose(sink._to_action_array(sink._last_action),current,atol=1e-12)
        sink.dispatch_action(sink._last_action)
        assert sink._collision_executor.filter.failure_count==0
    finally:
        sink.close()
