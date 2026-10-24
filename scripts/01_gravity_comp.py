## 通过 PD controller 可以使得任何的拖拽回到 home 位置
import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/tabletop.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
arm_dofs = [m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
arm_qpos = [m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
q_home = m.key("home").qpos[arm_qpos]
kp = np.array([100, 100, 100, 100, 30, 30, 30])
kd = np.array([10, 10, 10, 10, 3, 3, 3])

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        t0 = time.time()
        q = d.qpos[arm_qpos]
        qd = d.qvel[arm_dofs]
        # d.ctrl[:7] = d.qfrc_bias[arm_dofs]
        d.ctrl[:7] = kp * (q_home - q) - kd * qd + d.qfrc_bias[arm_dofs]
        d.ctrl[7] = 255
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t0)))