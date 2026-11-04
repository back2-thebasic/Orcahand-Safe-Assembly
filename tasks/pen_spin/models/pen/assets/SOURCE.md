Upstream: https://github.com/wuji-technology/wuji-mjlab
Commit: b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9
XML: src/wuji_mjlab/assets/objects/aruco_pen250/xmls/pen_tracking.xml (unmodified)
Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0; see LICENSE.wuji.
Ten tag textures copied from the user's v2026.10.22 release package. white.png retrieved from the pinned upstream commit.
Adapter changes: namespaced materials/geoms, body renamed pen, freejoint renamed pen_free, explicit inertia scaled to 30 g as upstream pen_cfg.py, explicit default contact parameters to avoid inheriting Orca defaults. Physical geometry is not rescaled.
