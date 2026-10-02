"""Independent discrete CBF-QP. No imports from teleop or legacy safety."""
import logging
import time
import numpy as np
from .collision import finite_difference_gradients
from .config import FilterConfig

logger = logging.getLogger(__name__)


def cbf_constraints(distances, gradients, config):
    """G delta >= -eta (d - d_safe). All distances in metres."""
    h = np.asarray(distances, dtype=float) - config.safe_distance
    return np.asarray(gradients, dtype=float), -config.cbf_eta * h


class CBFSafetyFilter:
    def __init__(self, model, low, high, max_step, config=None):
        import osqp
        from scipy import sparse
        self.model = model
        self.config = config or FilterConfig()
        self.low, self.high, self.max_step = [np.asarray(x,dtype=float).copy() for x in (low,high,max_step)]
        if (self.low.ndim != 1 or self.low.shape != self.high.shape or self.low.shape != self.max_step.shape
            or not all(np.all(np.isfinite(x)) for x in (self.low,self.high,self.max_step))
            or np.any(self.low >= self.high) or np.any(self.max_step <= 0)):
            raise ValueError('Invalid joint/step limits')
        self.n = self.low.size
        self.previous_safe_q = None
        self.failure_count = 0
        self._solver = osqp.OSQP()
        self._sparse = sparse
        self._active_names = None
        self._last_delta = np.zeros(self.n)

    def _vector(self, q):
        a=np.asarray(q,dtype=float)
        if a.shape!=(self.n,) or not np.all(np.isfinite(a)):
            raise ValueError('Expected finite joint position vector')
        return a

    def reset(self, q):
        q=self._vector(q)
        if np.any(q<self.low) or np.any(q>self.high):
            raise ValueError('Initial state violates joint limits')
        ds=self.model.distances(q)
        if not ds or any(d.collision or d.distance<self.config.safe_distance for d in ds):
            raise ValueError('Initial state violates collision safety margin; cannot initialize safe fallback')
        self.previous_safe_q=q.copy()
        self._last_delta[:]=0
        self.failure_count=0

    def filter(self, q_current, q_nominal):
        if self.previous_safe_q is None:
            raise RuntimeError('reset(valid_initial_state) required before filtering')
        started=time.perf_counter()
        info={'solver_status':'not_run','solver_iterations':0,'solver_time_ms':0.,
              'distance_time_ms':0.,'gradient_time_ms':0.,'num_active_constraints':0,
              'min_distance_before':None,'min_distance_after':None,'closest_pair':None,
              'command_backtracked':False,'command_step_scale':1.,'backtracking_steps':0,
              'fallback':False,'failure_count':self.failure_count}
        nominal=None
        try:
            current=self._vector(q_current);nominal=self._vector(q_nominal)
            t=time.perf_counter();before=self.model.distances(current)
            info['distance_time_ms']=(time.perf_counter()-t)*1000
            if not before or not all(np.isfinite(d.distance) for d in before):
                raise ValueError('Invalid distance measurements')
            nearest=min(before,key=lambda d:d.distance)
            active=[d for d in before if d.distance<self.config.activation_distance]
            names=tuple(d.pair_name for d in active)
            info.update(min_distance_before=nearest.distance,closest_pair=nearest.pair_name,
                        num_active_constraints=len(active),active_pairs=list(names),
                        pair_distances_before={d.pair_name:d.distance for d in before},
                        lower_bound_pairs=[d.pair_name for d in before if d.is_lower_bound])
            t=time.perf_counter()
            gradients=finite_difference_gradients(self.model,current,names,self.config.finite_difference_epsilon)
            info['gradient_time_ms']=(time.perf_counter()-t)*1000
            G,b=cbf_constraints([d.distance for d in active],gradients,self.config)
            lower=np.maximum(self.low-current,-self.max_step)
            upper=np.minimum(self.high-current,self.max_step)
            if np.any(lower>upper):raise ValueError('Infeasible joint/step bounds')
            # Scale CBF rows to keep solver tolerances in joint-sized units.
            scale=np.maximum(np.linalg.norm(G,axis=1),1e-6)
            A=np.vstack([np.eye(self.n),G/scale[:,None]])
            l=np.r_[lower,b/scale];u=np.r_[upper,np.full(len(active),np.inf)]
            # Explicit dense sparsity (including zeros) permits values-only updates.
            rows=A.shape[0]
            mat=self._sparse.csc_matrix((A.ravel(order='F'),np.tile(np.arange(rows),self.n),
                                         np.arange(self.n+1)*rows),shape=A.shape)
            t=time.perf_counter()
            if names!=self._active_names:
                import osqp
                self._solver=osqp.OSQP()
                self._solver.setup(P=self._sparse.eye(self.n,format='csc')*(1+2*self.config.regularization),
                                   q=-(nominal-current),A=mat,l=l,u=u,verbose=False,
                                   warm_starting=True,eps_abs=self.config.solver_tolerance,
                                   eps_rel=self.config.solver_tolerance,max_iter=self.config.max_iterations,
                                   polishing=False)
                self._active_names=names
            else:
                self._solver.update(q=-(nominal-current),Ax=mat.data,l=l,u=u)
            self._solver.warm_start(x=self._last_delta)
            result=self._solver.solve(raise_error=False)
            info.update(solver_time_ms=(time.perf_counter()-t)*1000,
                        solver_status=result.info.status,solver_iterations=int(result.info.iter))
            if result.info.status_val!=1 or result.x is None or not np.all(np.isfinite(result.x)):
                raise ValueError(f'QP {result.info.status}')
            delta=result.x
            if (np.any(delta<lower-1e-6) or np.any(delta>upper+1e-6)
                or np.any(G@delta < b-1e-8)):
                raise ValueError('QP solution fails residual check')
            # Roundoff clipping only, followed by CBF recheck.
            delta=np.clip(delta,lower,upper)
            if np.any(G@delta<b-1e-8):raise ValueError('Clipped QP violates CBF')
            safe=current+delta
            t=time.perf_counter();after=self.model.distances(safe)
            info['distance_time_ms']+=(time.perf_counter()-t)*1000
            info['min_distance_after']=min(d.distance for d in after)
            # Linearization is local. Validate every pair at the command, including
            # inactive pairs a large nominal step may newly approach.
            if any(not np.isfinite(d.distance) or d.collision or d.distance<self.config.safe_distance-1e-8 for d in after):
                info['rejected_min_distance_after']=info['min_distance_after']
                accepted=False
                # A local linearization may approve an endpoint too far away.
                # Shorten that same direction; never bypass endpoint or CBF checks.
                for attempt in range(1,self.config.nonlinear_backtracking_steps+1):
                    info['backtracking_steps']=attempt
                    scale=2.**(-attempt)
                    smaller=delta*scale
                    # Scaling is not necessarily feasible when measured state is
                    # outside a limit or margin and the CBF requires recovery.
                    if (np.any(smaller<lower-1e-8) or np.any(smaller>upper+1e-8)
                        or np.any(G@smaller<b-1e-8)):
                        continue
                    candidate=current+smaller
                    t=time.perf_counter();ds=self.model.distances(candidate)
                    info['distance_time_ms']+=(time.perf_counter()-t)*1000
                    if (ds and all(np.isfinite(d.distance) and not d.collision
                                   and d.distance>=self.config.safe_distance-1e-8 for d in ds)):
                        safe=candidate;delta=smaller
                        info.update(command_backtracked=True,command_step_scale=scale,
                                    min_distance_after=min(d.distance for d in ds))
                        accepted=True
                        break
                if not accepted:raise ValueError('nonlinear_command_margin_violation')
            self.previous_safe_q=safe.copy()
            self._last_delta=delta.copy()
        except Exception as exc:
            self.failure_count+=1
            safe=self.previous_safe_q.copy()
            info.update(fallback=True,failure_count=self.failure_count,error=str(exc))
            self._last_delta[:]=0
            if info['solver_status']=='not_run':info['solver_status']='error'
            # Min-after must refer to returned fallback, not rejected candidate.
            info.setdefault('rejected_min_distance_after',info['min_distance_after'])
            try:info['min_distance_after']=min(d.distance for d in self.model.distances(safe))
            except Exception:info['min_distance_after']=None
            if self.failure_count==1 or self.failure_count%100==0:
                logger.warning('COLLISION SAFETY FALLBACK #%d: %s',self.failure_count,exc)
        info.update(q_current=np.asarray(q_current).tolist(),q_nominal=np.asarray(q_nominal).tolist(),
                    q_safe=safe.tolist(),intervention_norm=float(np.linalg.norm(safe-nominal)) if nominal is not None else None,
                    total_safety_time_ms=(time.perf_counter()-started)*1000)
        return safe,info
