# Safety Layer: Four OFF/ON Motion Video Demos

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

Compare four motions in MuJoCo with the Safety Layer disabled (OFF) and enabled (ON), showing finger motion and safety interventions.

## Motion 1

- Motion: thumb–index pinch.
- Safety OFF: the thumb and index finger make full contact; collision occurs.
- Safety ON: the thumb and index finger approach each other while retaining a small gap.

Files: `motion1-OFF.mov`, `motion1-ON.mov`.

<!-- Insert Motion 1 OFF/ON videos here. -->

## Motion 2

- Motion: thumb–middle pinch.
- Safety OFF: the thumb and middle finger make full contact; collision occurs.
- Safety ON: the thumb and middle finger approach each other while retaining a small gap.

Files: `motion2-OFF.mov`, `motion2-ON.mov`.

<!-- Insert Motion 2 OFF/ON videos here. -->

## Motion 3

- Motion: thumb–index–middle three-finger pinch.
- Safety OFF: the thumb and index finger make full contact. The thumb and middle finger already cannot reach contact because of retargeting limitations.
- Safety ON: the three fingers remain separated, with small gaps at the end of the motion.

Files: `motion3-OFF.mov`, `motion3-ON.mov`.

<!-- Insert Motion 3 OFF/ON videos here. -->

## Motion 4

- Motion: index–middle contact.

This trajectory is synthetic. It was intended to produce crossed fingers, but MuJoCo's contact forces cause contact and compression rather than allowing the fingers to pass through each other.

- Safety OFF: the distal interphalangeal (DIP) regions make full contact, while the proximal interphalangeal (PIP) regions nearly touch.
- Safety ON: both distal and proximal regions maintain separation throughout the motion.

Files: `motion4-OFF.mov`, `motion4-ON.mov`.

<!-- Insert Motion 4 OFF/ON videos here. -->
