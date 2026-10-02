import numpy as np
import pytest
import mujoco
from orca_sim import OrcaHandRight
from orca_safety_v2.adapter import build_sim_model
from orca_safety_v2.collision import finite_difference_gradients


@pytest.fixture
def setup():
    env = OrcaHandRight(version='v1'); env.reset()
    args = build_sim_model(env)
    yield env, args
    env.close()


def test_open_hand_and_explicit_pairs(setup):
    env, (model, cfg, low, high, step, adr) = setup
    ds = model.distances(env.data.qpos[adr])
    assert len(ds) == 108
    assert min(d.distance for d in ds) > .005
    assert not any(d.collision for d in ds)
    for name in model.pair_indices:
        a,b = name.split('__')
        assert a.split('_')[1] != b.split('_')[1]
    np.testing.assert_allclose(step, 1.)


def test_fk_mapping_matches_mujoco(setup):
    env,(model,cfg,low,high,step,adr)=setup
    rng=np.random.default_rng(7)
    for _ in range(3):
        q=rng.uniform(low,high)
        env.data.qpos[adr]=q; mujoco.mj_forward(env.model,env.data)
        uq=model._update(q)
        model.pin.updateFramePlacements(model.model,model.data)
        palm=env.model.body('right_palm').id
        origin=env.data.xpos[palm]; rot=env.data.xmat[palm].reshape(3,3)
        pf=model.data.oMf[model.model.getFrameId('right_palm')]
        for name in ['right_index_ip','right_middle_pp','right_thumb_dp','right_ring_ip','right_pinky_ip']:
            body=env.model.body(name).id
            mj=rot.T@(env.data.xpos[body]-origin)
            pose=model.data.oMf[model.model.getFrameId(name)]
            pp=pf.rotation.T@(pose.translation-pf.translation)
            np.testing.assert_allclose(pp,mj,atol=2e-6)


def test_gradient_prediction_and_order(setup):
    env,(model,*rest)=setup
    q=env.data.qpos[rest[-1]].copy()
    names=['right_index_ip__right_middle_ip','right_thumb_dp__right_index_ip']
    g=finite_difference_gradients(model,q,names)
    assert g.shape==(2,17)
    perturb=np.random.default_rng(42).normal(size=17)*1e-6
    d=np.array([x.distance for x in model.distances(q,names)])
    actual=np.array([x.distance for x in model.distances(q+perturb,names)])-d
    np.testing.assert_allclose(g@perturb,actual,atol=1e-9,rtol=.02)
    assert np.linalg.norm(g)>1e-3
    assert np.max(abs(g[:,0]))<1e-6  # common wrist rotation cannot change distance


def test_broadphase_and_sparse_gradients_match_exact(setup):
    env,(model,cfg,low,high,step,adr)=setup
    q=env.data.qpos[adr].copy()
    exact=model.distances(q)
    model.broadphase_distance=cfg.activation_distance
    fast=model.distances(q)
    assert min(x.distance for x in fast)==pytest.approx(min(x.distance for x in exact))
    for a,b in zip(exact,fast):
        if a.distance<cfg.activation_distance:
            assert not b.is_lower_bound and a.distance==pytest.approx(b.distance)
        if b.is_lower_bound:assert b.distance<=a.distance+1e-10
    names=[d.pair_name for d in fast if d.distance<cfg.activation_distance]
    optimized=finite_difference_gradients(model,q,names)
    columns=model.gradient_columns;model.gradient_columns=None
    brute=finite_difference_gradients(model,q,names);model.gradient_columns=columns
    np.testing.assert_allclose(optimized,brute,atol=2e-8)


def test_fcl_known_box_gap_and_penetration(tmp_path):
    from orca_safety_v2.collision import FCLDistanceModel
    path=tmp_path/'boxes.urdf'
    path.write_text('''<robot name="boxes"><link name="world"/>
    <link name="a"><collision><geometry><box size="0.01 0.01 0.01"/></geometry></collision></link>
    <link name="b"><collision><geometry><box size="0.01 0.01 0.01"/></geometry></collision></link>
    <joint name="qa" type="prismatic"><parent link="world"/><child link="a"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    <joint name="qb" type="prismatic"><parent link="world"/><child link="b"/><origin xyz="0.02 0 0"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>''')
    model=FCLDistanceModel(path,[tmp_path],['qa','qb'],[0,0],[['a','b']])
    assert model.distances([0,0])[0].distance==pytest.approx(.01,abs=1e-8)
    np.testing.assert_allclose(finite_difference_gradients(model,np.zeros(2),['a__b']),[[-1,1]],atol=1e-6)
    hit=model.distances([0,-.011])[0]
    assert hit.collision and hit.distance<=0
