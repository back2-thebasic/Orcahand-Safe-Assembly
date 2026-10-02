# Safety Layer：四组动作 OFF/ON Demo Video

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

对比四组动作在 MuJoCo 中关闭（OFF）与开启（ON）Safety Layer 时的表现，用于直观观察手指运动与安全层的干预效果


## Motion 1

- 动作说明：拇指-食指捏和
- Safety OFF：拇指和食指完全接触，发生collision
- Safety ON：拇指和食指几乎接触，保持微小距离

对应文件：`motion1-OFF.mov`、`motion1-ON.mov`

<!-- 在此添加 Motion 1 的 OFF/ON 视频展示。 -->

## Motion 2

- 动作说明：拇指-中指捏和
- Safety OFF：拇指和中指完全接触，发生collision
- Safety ON：拇指和中指几乎接触，保持微小距离

对应文件：`motion2-OFF.mov`、`motion2-ON.mov`

<!-- 在此添加 Motion 2 的 OFF/ON 视频展示。 -->

## Motion 3

- 动作说明：拇指-食指-中指 三指捏和
- Safety OFF：拇指和食指完全接触，拇指和中指由于retarget的限制原本就无法接触
- Safety ON：三指未接触，并在动作结束时保持微小距离

对应文件：`motion3-OFF.mov`、`motion3-ON.mov`

<!-- 在此添加 Motion 3 的 OFF/ON 视频展示。 -->

## Motion 4

- 动作说明：食指-中指接触

（这条动作轨迹是合成的，原本想要合成食指和中指交叉的动作，但是由于 MuJoCo 自身的接触力，实际表现为两指接触、挤压，而不能完全穿过形成交叉）

- Safety OFF：两指远端指间关节(dip)完全接触，近端指间关节(pip)几乎接触
- Safety ON：远端指间关节(dip)和近端指间关节(pip)始终保持一定距离

对应文件：`motion4-OFF.mov`、`motion4-ON.mov`

<!-- 在此添加 Motion 4 的 OFF/ON 视频展示。 -->
