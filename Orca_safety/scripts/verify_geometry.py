"""Passive five-pose verification. Never filters or sends hardware commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from orca_sim import OrcaHandRight
from orca_safety_v2.adapter import build_sim_model
from orca_safety_v2.collision import finite_difference_gradients


def make_poses(model, q0, low, high):
    names=list(model.joint_names)
    poses={'open':q0.copy()}
    fist=q0.copy()
    for i,name in enumerate(names):
        if name.endswith(('_mcp','_pip')) and 'thumb' not in name: fist[i]=1.2
    poses['fist']=fist
    for finger in ['index','middle']:
        pair=f'right_thumb_dp__right_{finger}_ip'
        indices=[i for i,n in enumerate(names) if n.startswith(('right_thumb_',f'right_{finger}_'))]
        def obj(x):
            q=q0.copy();q[indices]=x
            d=model.distances(q,[pair])[0]
            return abs(d.distance-.001) + (1 if d.collision else 0)
        res=differential_evolution(obj,list(zip(low[indices],high[indices])),seed=12,
                                   maxiter=35,popsize=6,polish=False)
        q=q0.copy();q[indices]=res.x;poses[f'thumb_{finger}_pinch']=q
    # Explicitly search for an intersecting mesh pose, with safety OFF.
    rng=np.random.default_rng(12)
    pair='right_index_ip__right_middle_ip'
    indices=[i for i,n in enumerate(names) if n.startswith(('right_index_','right_middle_'))]
    for _ in range(2000):
        q=q0.copy();q[indices]=rng.uniform(low[indices],high[indices])
        if model.distances(q,[pair])[0].collision:
            poses['intentional_collision']=q;break
    if 'intentional_collision' not in poses: raise RuntimeError('No collision fixture found')
    return poses


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/passive'))
    p.add_argument('--render',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env=OrcaHandRight(version='v1');env.reset();m,c,low,high,step,adr=build_sim_model(env)
    poses=make_poses(m,env.data.qpos[adr].copy(),low,high)
    rows=[]
    renderer=mujoco.Renderer(env.model,480,640) if args.render else None
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(env.model,camera)
    for frame,(name,q) in enumerate(poses.items()):
        env.data.qpos[adr]=q;env.data.qvel[:]=0;mujoco.mj_forward(env.model,env.data)
        t=time.perf_counter();ds=m.distances(q);ms=(time.perf_counter()-t)*1000
        closest=min(ds,key=lambda d:d.distance)
        row={'frame':frame,'pose':name,'filter':'OFF','q':q.tolist(),'joint_names':list(m.joint_names),
             'minimum_distance':closest.distance,'minimum_pair':closest.pair_name,
             'all_active_pair_distances':[asdict(d) for d in ds], 'distance_time_ms':ms}
        rows.append(row)
        if renderer:
            import PIL.Image
            renderer.update_scene(env.data,camera=camera)
            PIL.Image.fromarray(renderer.render()).save(args.output/f'{name}.png')
        print(name,closest.distance,closest.pair_name,'collisions',sum(d.collision for d in ds))
    (args.output/'poses.json').write_text(json.dumps({k:v.tolist() for k,v in poses.items()},indent=2))
    (args.output/'distances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # Selected non-degenerate pairs, numerical sign/prediction test.
    q=poses['open'];pairs=['right_thumb_dp__right_index_ip','right_index_ip__right_middle_ip']
    g=finite_difference_gradients(m,q,pairs);dq=np.random.default_rng(42).normal(size=len(q))*1e-6
    actual=np.array([d.distance for d in m.distances(q+dq,pairs)])-np.array([d.distance for d in m.distances(q,pairs)])
    (args.output/'gradient.json').write_text(json.dumps({'pairs':pairs,'predicted':(g@dq).tolist(),'actual':actual.tolist(),'max_error':float(max(abs(g@dq-actual)))},indent=2))
    if renderer:renderer.close()
    env.close()

if __name__=='__main__':main()
