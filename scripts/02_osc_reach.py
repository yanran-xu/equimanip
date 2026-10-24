# TCP follows the green target site. 
# Move the target with the mouse in the viewer (double-click it, then Ctrl + right-drag), 
# reading it from d.site_xpos[m.site("target").id] each step.

import time
import mujoco
import mujoco.viewer
import numpy as np
from equimanip.osc import OSC

m = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/tabletop.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.mj_forward(m,d)

controller = OSC(m, d)
tcp = m.site("tcp").id
target_id = m.site("target").id
target_quat = np.zeros(4)
mujoco.mju_mat2Quat(target_quat, d.site_xmat[tcp])

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        t0 = time.time()
        target_pos = d.site_xpos[target_id] + np.array([0,0,0.05])
        d.ctrl[:7] = controller(target_pos, target_quat)
        d.ctrl[7] = 255
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t0)))