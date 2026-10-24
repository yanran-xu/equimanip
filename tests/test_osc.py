import mujoco
import numpy as np
from equimanip.osc import OSC

def test_reaches_targets():
    m = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/tabletop.xml")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    osc = OSC(m, d)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, d.site_xmat[osc.site])
    start = d.site_xpos[osc.site].copy()
    for target in [start + [0, 0.15, -0.1], start + [0.1, -0.1, -0.2], start]:
        for _ in range(1500):          # 3 s of sim time
            d.ctrl[:7] = osc(target, quat)
            d.ctrl[7] = 255
            mujoco.mj_step(m, d)
        assert np.linalg.norm(target - d.site_xpos[osc.site]) < 1e-3   # under 1 mm