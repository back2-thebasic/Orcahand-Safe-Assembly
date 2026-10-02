import numpy as np
import pytest
from orca_safety_v2 import CBFSafetyFilter, FilterConfig, CollisionDistance
from orca_safety_v2.filter import cbf_constraints


class LinearGeometry:
    def distances(self,q,pair_names=None):
        all=[CollisionDistance('a',float(q[0]+.01),q[0]+.01<=0),
             CollisionDistance('b',float(q[1]+.01),q[1]+.01<=0)]
        return all if pair_names is None else [next(d for d in all if d.pair_name==n) for n in pair_names]


def build():
    f=CBFSafetyFilter(LinearGeometry(),np.array([-.1,-.1]),np.array([.1,.1]),np.array([.02,.02]))
    f.reset(np.zeros(2));return f


def test_safe_unchanged_and_warm_start():
    f=build()
    for _ in range(2):
        q,d=f.filter(np.zeros(2),np.array([.001,.001]))
        np.testing.assert_allclose(q,[.001,.001],atol=1e-7)
        assert d['solver_status']=='solved' and not d['fallback']


def test_multiple_constraints_and_sign():
    f=build();q,d=f.filter(np.zeros(2),np.array([-.05,-.04]))
    np.testing.assert_allclose(q,[-.001,-.001],atol=2e-7)
    assert d['num_active_constraints']==2
    G,b=cbf_constraints([.01],[[1.,0]],FilterConfig())
    np.testing.assert_allclose(b,[-.001]);assert (G@q>=b-1e-7).all()


def test_joint_and_step_limits():
    f=build();q,d=f.filter(np.array([.095,.095]),np.array([3.,3.]))
    assert np.all(q<=.1) and np.all(q>=.095)
    f=build();q,d=f.filter(np.zeros(2),np.array([3.,3.]))
    assert np.all(q<=.020001)


def test_at_boundary_cannot_approach():
    f=build();q,d=f.filter(np.array([-.005,-.005]),np.array([-.02,-.02]))
    assert np.all(q>=-.005-1e-8)


@pytest.mark.parametrize('bad',[np.array([np.nan,0]),np.array([1.])])
def test_invalid_input_fallback(bad):
    f=build();q,d=f.filter(np.zeros(2),bad)
    np.testing.assert_array_equal(q,[0,0]);assert d['fallback'] and f.failure_count==1


def test_infeasible_qp_fallback():
    f=build()
    # Actual position is deeply unsafe and cannot reach limits within max_step.
    q,d=f.filter(np.array([-.2,-.2]),np.array([.02,.02]))
    np.testing.assert_array_equal(q,[0,0]);assert d['fallback']


def test_solver_exception_fallback(monkeypatch):
    f=build();f.filter(np.zeros(2),np.zeros(2))
    def fail(**kwargs):raise RuntimeError('injected')
    monkeypatch.setattr(f._solver,'solve',fail)
    q,d=f.filter(np.zeros(2),np.array([.01,.01]))
    np.testing.assert_allclose(q,[0,0],atol=1e-9);assert d['fallback']


def test_real_infeasible_constraints():
    class Impossible(LinearGeometry):
        def distances(self,q,pair_names=None):
            return [CollisionDistance('fixed',.006 if q[0]==0 else .001,False)]
    f=CBFSafetyFilter(Impossible(),[-1,-1],[1,1],[.1,.1]);f.reset([0,0])
    q,d=f.filter([.5,.5],[.6,.6])
    assert d['fallback'] and 'infeasible' in d['solver_status']
    np.testing.assert_array_equal(q,[0,0])


def test_reset_rejects_unsafe_and_must_be_explicit():
    f=build()
    with pytest.raises(ValueError):f.reset([-.01,0])
    f.previous_safe_q=None
    with pytest.raises(RuntimeError):f.filter([0,0],[0,0])


def test_inactive_pair_large_jump_is_not_silently_sent():
    class Nonlinear:
        def distances(self,q,pair_names=None):
            d=(q[0]-.1)**2+.001
            return [CollisionDistance('curve',d,False)]
    f=CBFSafetyFilter(Nonlinear(),[-1],[1],[1],FilterConfig(activation_distance=.006))
    f.reset([0])
    q,d=f.filter([0],[.1])
    assert not d['fallback'] and d['command_backtracked']
    assert 0 < q[0] < .1 and d['command_step_scale'] < 1
    assert f.model.distances(q)[0].distance >= f.config.safe_distance-1e-8
    np.testing.assert_array_equal(f.previous_safe_q,q)


def test_backtracking_can_be_disabled_for_original_behavior():
    class Nonlinear:
        def distances(self,q,pair_names=None):
            return [CollisionDistance('curve',(q[0]-.1)**2+.001,False)]
    f=CBFSafetyFilter(Nonlinear(),[-1],[1],[1],FilterConfig(
        activation_distance=.006,nonlinear_backtracking_steps=0))
    f.reset([0]);q,d=f.filter([0],[.1])
    assert d['fallback'] and d['error']=='nonlinear_command_margin_violation'
    np.testing.assert_array_equal(q,[0])


def test_backtracking_keeps_fallback_when_no_safe_progress_exists():
    class Discontinuous:
        def distances(self,q,pair_names=None):
            return [CollisionDistance('blocked',.02 if q[0]==0 else .001,False)]
    f=CBFSafetyFilter(Discontinuous(),[-1],[1],[1],FilterConfig(activation_distance=.015))
    f.reset([0]);q,d=f.filter([0],[.1])
    assert d['fallback'] and not d['command_backtracked']
    assert d['backtracking_steps']==8
    np.testing.assert_array_equal(q,[0])
