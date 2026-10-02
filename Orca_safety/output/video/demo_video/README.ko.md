# Safety Layer: 네 가지 동작의 OFF/ON 데모 영상

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

MuJoCo에서 Safety Layer를 끈 경우(OFF)와 켠 경우(ON)의 네 가지 동작을 비교하여 손가락 움직임과 안전층의 개입을 관찰합니다.

## Motion 1

- 동작: 엄지–검지 집기.
- Safety OFF: 엄지와 검지가 완전히 접촉하며 충돌이 발생합니다.
- Safety ON: 엄지와 검지가 거의 접촉할 정도로 접근하지만 작은 간격을 유지합니다.

파일: `motion1-OFF.mov`, `motion1-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/a43aef80-2c4b-40b7-8f38-0082a0ad63db

### Safety ON

https://github.com/user-attachments/assets/691d364a-c1b2-442b-b365-47c8ae855ce6

## Motion 2

- 동작: 엄지–중지 집기.
- Safety OFF: 엄지와 중지가 완전히 접촉하며 충돌이 발생합니다.
- Safety ON: 엄지와 중지가 거의 접촉할 정도로 접근하지만 작은 간격을 유지합니다.

파일: `motion2-OFF.mov`, `motion2-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/3eabf2e1-226e-45e6-92ef-e90b1ad611d2

### Safety ON

https://github.com/user-attachments/assets/d80fd20a-fcc4-4c40-be6d-1ab639fa5535

## Motion 3

- 동작: 엄지–검지–중지의 세 손가락 집기.
- Safety OFF: 엄지와 검지는 완전히 접촉합니다. 엄지와 중지는 retargeting의 제한 때문에 원래부터 접촉할 수 없습니다.
- Safety ON: 세 손가락이 접촉하지 않으며 동작이 끝날 때 작은 간격을 유지합니다.

파일: `motion3-OFF.mov`, `motion3-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/478cfb7f-b3b8-45f0-9f52-8d99d600a1e1

### Safety ON

https://github.com/user-attachments/assets/b359125d-be57-43c8-8a2a-4f864285b22c

## Motion 4

- 동작: 검지–중지 접촉.

이 궤적은 합성 입력입니다. 원래 검지와 중지의 교차를 만들려 했지만, MuJoCo 자체 접촉력으로 인해 손가락이 서로 통과하지 못하고 접촉 및 압박하는 동작이 나타납니다.

- Safety OFF: 원위지간관절(DIP) 부위가 완전히 접촉하고 근위지간관절(PIP) 부위는 거의 접촉합니다.
- Safety ON: 원위 및 근위 부위가 동작 전체에서 일정한 간격을 유지합니다.

파일: `motion4-OFF.mov`, `motion4-ON.mov`.

### Safety OFF

https://github.com/user-attachments/assets/f5c81db7-299b-4e46-91d7-1539c19ba6bd

### Safety ON

https://github.com/user-attachments/assets/67ad9e23-8ceb-4819-b793-34da7942d8b4
