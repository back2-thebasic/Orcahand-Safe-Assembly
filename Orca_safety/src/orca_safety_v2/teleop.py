"""Optional simulation integration and structured per-frame diagnostics."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import numpy as np
from .adapter import build_sim_model
from .filter import CBFSafetyFilter
from .paths import relative_path

logger=logging.getLogger(__name__)


def json_finite(obj):
    if isinstance(obj,dict):return {k:json_finite(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [json_finite(v) for v in obj]
    if isinstance(obj,(float,np.floating)) and not np.isfinite(obj):return None
    return obj


class SimulationCollisionSafety:
    def __init__(self, env, config_path=None, log_path=None, monitor_only=False, urdf_path=None):
        self.env=env
        self.model,self.config,low,high,max_step,self.addresses=build_sim_model(env,config_path,urdf_path)
        self.model.broadphase_distance=self.config.activation_distance
        self.monitor_only=monitor_only
        self.filter=None if monitor_only else CBFSafetyFilter(self.model,low,high,max_step,self.config)
        current=env.data.qpos[self.addresses].copy()
        if self.filter:self.filter.reset(current)
        self.frame=0
        self.last_warning=0.
        if log_path is None:
            log_path=Path(__file__).resolve().parents[2]/'output/teleop'/f'{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl'
        path=Path(log_path);path.parent.mkdir(parents=True,exist_ok=True)
        self.log=path.open('x')
        self._write({'event':'start','software':'orca_safety_v2','robot':'v1_right','mode':'monitor' if monitor_only else 'CBF-QP',
                     'joint_names':list(self.model.joint_names),'units':{'angle':'rad','distance':'m','time':'s'},
                     'config':vars(self.config),'low':low.tolist(),'high':high.tolist(),'max_step':max_step.tolist(),
                     'reference_offsets':self.model.offsets.tolist(),'pairs':list(self.model.pair_indices)})
        logger.info('COLLISION SAFETY %s | %s','MONITOR' if monitor_only else 'READY',relative_path(path))

    def _write(self,row):
        self.log.write(json.dumps(json_finite(row),allow_nan=False)+'\n');self.log.flush()

    def dispatch(self, nominal):
        current=self.env.data.qpos[self.addresses].copy()
        if self.filter:
            safe,info=self.filter.filter(current,nominal)
        else:
            before=self.model.distances(current);nearest=min(before,key=lambda d:d.distance)
            safe=np.asarray(nominal,dtype=float)
            info={'q_current':current.tolist(),'q_nominal':safe.tolist(),'q_safe':safe.tolist(),
                  'min_distance_before':nearest.distance,'closest_pair':nearest.pair_name,
                  'pair_distances_before':{d.pair_name:d.distance for d in before},
                  'lower_bound_pairs':[d.pair_name for d in before if d.is_lower_bound],
                  'solver_status':'monitor_only','intervention_norm':0.,'fallback':False}
        self.env.step(safe)
        measured=self.model.distances(self.env.data.qpos[self.addresses])
        minimum=min(measured,key=lambda d:d.distance)
        info.update(event='frame',frame=self.frame,timestamp=datetime.now(timezone.utc).isoformat(),
                    sim_time=float(self.env.data.time),min_distance_measured=minimum.distance,
                    measured_collision=any(d.collision for d in measured),
                    measured_margin_violation=minimum.distance<self.config.safe_distance,
                    q_measured=self.env.data.qpos[self.addresses].tolist())
        self._write(info);self.frame+=1
        now=time.monotonic()
        if (info['fallback'] or info['measured_margin_violation']) and now-self.last_warning>1:
            state=('FALLBACK: '+info.get('error','unknown')) if info['fallback'] else 'MEASURED MARGIN VIOLATION'
            logger.warning('COLLISION SAFETY | %s | min=%.2f mm | %s',state,minimum.distance*1000,minimum.pair_name)
            self.last_warning=now
        return info

    def close(self):
        self.log.close()
