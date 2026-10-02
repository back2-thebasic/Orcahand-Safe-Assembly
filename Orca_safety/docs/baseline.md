# Original System Interfaces and Safety Layer Integration

[English](baseline.md) | [中文](baseline.zh.md) | [한국어](baseline.ko.md)

The current robot model is the **v1 right hand**; v2 is the safety software version. See [implementation.md](implementation.md) for implementation and validation.

## Action Pipeline

```text
Hand landmarks → Retargeter → OrcaJointPositions → actions_q
→ Simulation executor → Safety Layer (optional) → MuJoCo env.step
```

The retargeter outputs absolute angles for 17 joints in degrees. The simulation executor orders them by joint name and converts them to radians before passing them to the safety layer.

- `q_nominal`: original target.
- `q_current`: joint angles read from the actual MuJoCo state.
- `q_safe`: target returned by the safety layer and passed to the simulation.

The safety layer has not been integrated with the physical hand. OFF/ON comparisons use the same model, initial state, and nominal inputs.

## Joint Mapping

MuJoCo actuator order:

```text
wrist → thumb(mcp, abd, pip, dip)
→ index, middle, ring, pinky (abd, mcp, pip)
```

MuJoCo and Pinocchio joints are mapped by name. Actual joint positions are read through `jnt_qposadr`. URDF geometry calculations account for the reference offset: `q_URDF = q_MuJoCo − model.qpos0`. Angles must not be copied directly by array position.

## Geometry and Limit Sources

- **Collision geometry**: collision meshes from the v1 URDF, queried through Pinocchio/FCL.
- **Joint ranges**: intersection of MuJoCo `jnt_range` and `actuator_ctrlrange`.
- **Per-step motion limits**: URDF velocity limits × control period; currently 100 rad/s × 0.01 s = 1 rad/step. `max_step_rad` can impose a tighter limit.
