"""Exercise the unchanged retargeter with synthetic MediaPipe-format landmarks.

This is a reproducible integration test, not captured human tracking data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import orca_core
from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.retargeter import TargetPose
from orca_teleop.retargeting.constants import CALIBRATION_FRAMES
from orca_teleop.sim import OrcaHandSimSink
from orca_sim import OrcaHandRight
from orca_safety_v2.adapter import build_sim_model
from orca_safety_v2.paths import relative_path


def landmarks(curl, spread):
    points=np.zeros((21,3))
    for f,(x,y,length) in enumerate([(-.035,.025,.060),(-.025,.065,.080),(0,.075,.09),(.022,.07,.085),(.042,.055,.07)]):
        base=np.array([x,y,0.]);points[1+4*f]=base
        for segment in range(1,4):
            angle=curl*segment*.55
            lateral=(-.25 if f==0 else (f-2)*spread)
            direction=np.array([lateral,np.cos(angle),-np.sin(angle)])
            points[1+4*f+segment]=points[4*f+segment]+length/3*direction
    return points


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/replay/retarget-replay.npz'))
    p.add_argument('--steps',type=int,default=120);args=p.parse_args()
    workspace=Path(__file__).resolve().parents[2]
    sink=OrcaHandSimSink(version='v1');config=sink._resolve_retarget_model_path('v1')
    urdf=workspace/'orcahand_description/v1/models/urdf/orcahand_right.urdf'
    # Use the existing v1 baseline config, without optional fingertip overrides.
    retarget_config=workspace/'orca_adaptive_test/configs/baseline.yaml'
    r=AdaptiveAnalyticalRetargeter.from_paths(model_path=config,urdf_path=str(urdf),config_path=str(retarget_config))
    for _ in range(CALIBRATION_FRAMES):r.retarget(TargetPose(joint_positions=landmarks(0,.08),source='mediapipe'))
    env=OrcaHandRight(version='v1');model,*_=build_sim_model(env);env.close()
    joint_ids=[n.removeprefix('right_') for n in model.joint_names]
    qs=[];inputs=[]
    for phase in np.linspace(0,2*np.pi,args.steps):
        pose=landmarks(.7*(1-np.cos(phase)),.12*np.cos(phase))
        action=r.retarget(TargetPose(joint_positions=pose,source='mediapipe'))
        if action is None:raise RuntimeError('Calibration did not finish')
        qs.append(np.deg2rad(action.as_array(joint_ids)));inputs.append(pose)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,q_nominal=qs,landmarks=inputs,joint_names=model.joint_names)
    args.output.with_suffix('.json').write_text(json.dumps({'source':'synthetic MediaPipe-format landmarks; unchanged adaptive_analytical with existing v1 baseline.yaml',
        'retarget_config':relative_path(retarget_config),'calibration_frames':CALIBRATION_FRAMES,'frames':args.steps,
        'model_config':relative_path(config, Path(orca_core.__file__).resolve().parent),'urdf':relative_path(urdf),
        'path_bases':{'retarget_config':'orca_safety_v2 package root','urdf':'orca_safety_v2 package root','model_config':'installed orca_core package root'},
        'units':'q_nominal radians; landmarks metres','retargeter_sha256':hashlib.sha256((workspace/'orca_teleop/src/orca_teleop/retargeting/adaptive_analytical.py').read_bytes()).hexdigest()},indent=2))
    print(relative_path(args.output))

if __name__=='__main__':main()
