# OrcaHand Pen Spin

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)
本目录整理了转笔任务使用的模型、官方示范、转换数据和实验结果。

```text
pen_spin/
├── models/
│   ├── pen/              官方 MuJoCo 笔模型、纹理和来源信息
│   └── hand/             回放所需的 OrcaHand 网格资源
├── hardware/             笔的 STEP、3MF 打印文件及视觉标记资源
├── data/
│   ├── official/         40 条官方 demonstration 及索引、配方
│   └── converted/        20 条初次 retarget 到 OrcaHand 的转换结果
├── results/
│   ├── inspection-*     003/006 的检查结果和图片
│   └── optimized003-*   003 的各轮优化结果及优化前后对比索引
├── view.py               参考轨迹与自由动力学回放工具
├── portable_scene.py     回放使用的场景路径辅助工具
└── requirements.txt      回放所需的 Python 依赖
```

`data/converted` 保存初次转换结果；`results` 保存后续检查和优化产物。目前这些数据仍是候选参考轨迹，自由动力学中尚未成功完成转笔，未开始 RL 训练。

笔模型入口为 `models/pen/assets/pen_tracking.xml`，依赖同级目录结构中的 `models/pen/textures/`。这是官方原始 XML；官方加载代码另外将质量设为 30 g 并按比例缩放惯量，该步骤未固化到此 XML 中。
