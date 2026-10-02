# Safety Layer: Four OFF/ON Motion Video Demos

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

Compare four motions in MuJoCo with the Safety Layer disabled (OFF) and enabled (ON), showing finger motion and safety interventions.

## Motion 1

- Motion: thumb–index pinch.
- Safety OFF: the thumb and index finger make full contact; collision occurs.
- Safety ON: the thumb and index finger approach each other while retaining a small gap.

Files: `motion1-OFF.mov`, `motion1-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/a43aef80-2c4b-40b7-8f38-0082a0ad63db

### Safety ON

https://github.com/user-attachments/assets/691d364a-c1b2-442b-b365-47c8ae855ce6

## Motion 2

- Motion: thumb–middle pinch.
- Safety OFF: the thumb and middle finger make full contact; collision occurs.
- Safety ON: the thumb and middle finger approach each other while retaining a small gap.

Files: `motion2-OFF.mov`, `motion2-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/3eabf2e1-226e-45e6-92ef-e90b1ad611d2

### Safety ON

https://github.com/user-attachments/assets/d80fd20a-fcc4-4c40-be6d-1ab639fa5535

## Motion 3

- Motion: thumb–index–middle three-finger pinch.
- Safety OFF: the thumb and index finger make full contact. The thumb and middle finger already cannot reach contact because of retargeting limitations.
- Safety ON: the three fingers remain separated, with small gaps at the end of the motion.

Files: `motion3-OFF.mov`, `motion3-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/478cfb7f-b3b8-45f0-9f52-8d99d600a1e1

### Safety ON

https://github.com/user-attachments/assets/b359125d-be57-43c8-8a2a-4f864285b22c

## Motion 4

- Motion: index–middle contact.

This trajectory is synthetic. It was intended to produce crossed fingers, but MuJoCo's contact forces cause contact and compression rather than allowing the fingers to pass through each other.

- Safety OFF: the distal interphalangeal (DIP) regions make full contact, while the proximal interphalangeal (PIP) regions nearly touch.
- Safety ON: both distal and proximal regions maintain separation throughout the motion.

Files: `motion4-OFF.mov`, `motion4-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/f5c81db7-299b-4e46-91d7-1539c19ba6bd

### Safety ON

https://github.com/user-attachments/assets/67ad9e23-8ceb-4819-b793-34da7942d8b4
