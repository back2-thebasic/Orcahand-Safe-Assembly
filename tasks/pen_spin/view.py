"""Interactive replay of a saved probe. On macOS run with mjpython."""
import argparse
import json
import queue
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'data/demo_repair/scripts'))
from portable_scene import scene_signature
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--run',type=Path)
    g.add_argument('--catalog',type=Path,help='convert.py output directory (data/demo_repair/results/converted) or catalog.json')
    p.add_argument('--mode',choices=['dynamic','reference','hold'],default='hold')
    p.add_argument('--frame',type=int,default=0)
    a=p.parse_args()
    if a.catalog:
        path=a.catalog/'catalog.json' if a.catalog.is_dir() else a.catalog
        entries=[e for e in json.loads(path.read_text())['entries'] if not e.get('error')]
        for entry in entries:
            entry['run']=str((path.resolve().parent/entry['run']).resolve())
        if not entries:p.error('No converted clips')
    else:
        run=a.run or Path(__file__).parent/'data/demo_repair/results/trajectory/02_continuous_single_axis__clip_002'
        entries=[dict(run=str(run),clip=run.name)]
    index=0
    print('N/P next/previous | 1 reference | 2 static hold | 3 dynamic | SPACE pause | R reset | Q exit',flush=True)
    show(entries[index],a.mode,a.frame,index,len(entries),entries=entries)


def show(entry,mode,start_frame,index,total,entries=None):
    entries=entries or [entry]
    run=Path(entry['run'])
    # These official-pen clips share one model. Reuse ONE viewer for its lifetime:
    # Handle.close() only requests async UI shutdown on macOS, so immediately
    # launching another viewer races the mjpython UI dispatcher.
    scene_hash=scene_signature(run/'scene.xml')
    for candidate in entries:
        if scene_signature(Path(candidate['run'])/'scene.xml')!=scene_hash:
            raise ValueError('Catalog scenes differ; open each with --run instead of switching models in one viewer')
    m=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));d=mujoco.MjData(m)
    def load_clip(candidate):
        with np.load(Path(candidate['run'])/'reference.npz') as data:
            q=data['reference_qpos'].copy();obj=data['object_pose_w'].copy();dt=float(data['frame_dt'])
        if q.ndim!=2 or q.shape[1]!=m.nu or obj.shape!=(len(q),7) or not len(q):
            raise ValueError('Invalid clip dimensions')
        if not np.isfinite(q).all() or not np.isfinite(obj).all() or not np.isfinite(dt) or dt<=0:
            raise ValueError('Invalid clip values')
        return q,obj,dt
    q,obj,dt=load_clip(entry)
    if not 0<=start_frame<len(q):raise ValueError('Frame outside clip')
    joints=m.actuator_trnid[:,0];qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints]
    pq=m.joint('pen_free').qposadr[0];pv=m.joint('pen_free').dofadr[0]
    print(f'[{index+1}/{total}] {entry["clip"]} | {mode} | initial frame {start_frame}',flush=True)
    def reset():
        i=start_frame
        mujoco.mj_resetData(m,d);d.qpos[qa]=q[i];d.qpos[pq:pq+7]=obj[i];d.ctrl[:]=q[i]
        if mode=='dynamic' and i+1<len(q):
            d.qvel[va]=(q[i+1]-q[i])/dt;d.qvel[pv:pv+3]=(obj[i+1,:3]-obj[i,:3])/dt
            r=Rotation.from_quat(obj[i:i+2,[4,5,6,3]])
            d.qvel[pv+3:pv+6]=(r[0].inv()*r[1]).as_rotvec()/dt
        mujoco.mj_forward(m,d)
    reset()
    keys=queue.Queue();paused=False;k=start_frame;elapsed=0.
    with mujoco.viewer.launch_passive(m,d,key_callback=keys.put) as viewer:
        viewer.cam.lookat[:]=obj[start_frame,:3];viewer.cam.distance=.48
        viewer.cam.azimuth=90;viewer.cam.elevation=-55
        while viewer.is_running():
            start=time.monotonic()
            while not keys.empty():
                key=keys.get_nowait()
                if 65<=key<=90:key+=32
                if key==ord('q'):return 0,mode
                if key in (ord('n'),ord('p')):
                    index=(index+(1 if key==ord('n') else -1))%len(entries)
                    entry=entries[index];q,obj,dt=load_clip(entry);start_frame=0
                    with viewer.lock():
                        reset();k=0;elapsed=0.
                        viewer.cam.lookat[:]=obj[0,:3]
                    print(f'[{index+1}/{total}] {entry["clip"]} | {mode} | initial frame 0',flush=True)
                    continue
                if key in (ord('1'),ord('2'),ord('3')):
                    mode={ord('1'):'reference',ord('2'):'hold',ord('3'):'dynamic'}[key]
                    with viewer.lock():
                        reset();k=start_frame;elapsed=0.
                elif key==ord('r'):
                    with viewer.lock():
                        reset();k=start_frame;elapsed=0.
                elif key==32:paused=not paused
            with viewer.lock():
                if not paused and mode=='reference':
                    d.qpos[qa]=q[k];d.qpos[pq:pq+7]=obj[k];d.qvel[:]=0
                    mujoco.mj_forward(m,d)
                    k=k+1 if k+1<len(q) else start_frame
                elif not paused:
                    if mode=='dynamic' and k>=len(q):reset();k=start_frame;elapsed=0.
                    d.ctrl[:]=q[start_frame if mode=='hold' else k];d.xfrc_applied[:]=0;d.qfrc_applied[:]=0
                    mujoco.mj_step(m,d,nstep=round(dt/m.opt.timestep))
                    mujoco.mj_forward(m,d)
                    if mode=='dynamic':k+=1
                if not paused:elapsed+=dt
            label={'reference':'FORCED POSE - NOT manipulation success','hold':'FIXED hand targets - FREE pen','dynamic':'MOVING hand targets - FREE pen'}[mode]
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                f'{index+1}/{total} {entry["clip"]}',
                f'{label}\nFrame {min(k,len(q)-1)} | t={elapsed:.2f}s | {"PAUSED" if paused else "RUNNING"}\n'
                'N/P clips | 1 reference | 2 hold | 3 dynamic\nSPACE pause | R reset | Q exit')])
            viewer.sync()
            time.sleep(max(0,dt-(time.monotonic()-start)))
    return 0,mode


if __name__=='__main__':main()
