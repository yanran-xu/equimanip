# Operational Space Control (OSC) for the arm. The controller computes a torque command
import mujoco
import numpy as np


class OSC:
    def __init__(self, model, data, site="tcp", kp_pos=400.0, kp_rot=150.0, kp_null=10.0):
        self.m, self.d = model, data
        self.site = model.site(site).id
        self.dof = np.array([model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.kp = np.r_[np.full(3, kp_pos), np.full(3, kp_rot)]
        self.kd = 2.0 * np.sqrt(self.kp)  # why this value? (hint: critical damping)
        self.kp_null = kp_null
        self.kd_null = 2.0 * np.sqrt(kp_null)
        self.q_home = model.key("home").qpos[self.qadr].copy()

        # Jacobian and mass matrix buffers 缓冲区
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.M_full = np.zeros((model.nv, model.nv))

    def __call__(self, target_pos, target_quat):
        m, d = self.m, self.d
        dof = self.dof

        # TODO 1: Jacobian. mujoco.mj_jacSite(m, d, jacp, jacr, self.site) fills two 3 x nv arrays.
        #         Stack them into 6 x nv, then keep only the arm columns (self.dof).
        mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.site)
        J = np.vstack([self.jacp, self.jacr])[:, self.dof]

        # TODO 2: Mass matrix. mujoco.mj_fullM(m, d, M_full) fills an nv x nv array; take the arm block.
        #         Then Lam = inverse of (J M^-1 J^T). Use np.linalg.pinv(..., rcond=1e-3) near singularities.
        mujoco.mj_fullM(m, d, self.M_full)
        M = self.M_full[np.ix_(dof, dof)]
        M_inv = np.linalg.inv(M)
        Lam = np.linalg.pinv(J @ M_inv @ J.T, rcond=1e-3)


        # TODO 3: Pose error. Position: target - d.site_xpos[self.site].
        #         Orientation: current quat via mujoco.mju_mat2Quat(q, d.site_xmat[self.site]),
        #         then the error quat = target * conj(current) (mju_negQuat, mju_mulQuat),
        #         then convert to a 3D axis-angle vector with mujoco.mju_quat2Vel(out, err_quat, 1.0).
        e_pos = target_pos - d.site_xpos[self.site]
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site])
        cur_conj = np.zeros(4)
        mujoco.mju_negQuat(cur_conj, cur_quat)
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, target_quat, cur_conj)
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, err_quat, 1.0)

        e = np.r_[e_pos, e_rot]

        # TODO 4: tau = J^T Lam (kp*e - kd*J qdot)
        qd = d.qvel[dof]
        ee_vel = J @ qd
        F =  Lam @ (self.kp * e - self.kd * ee_vel)
        tau = J.T @ F


        # TODO 5: add the null-space posture torque, projected by N
        q = d.qpos[self.qadr]
        tau_null = self.kp_null * (self.q_home - q) - self.kd_null * qd
        J_bar = M_inv @ J.T @ Lam
        N = np.eye(len(dof)) - J.T @ J_bar.T
        tau += N @ tau_null


        # TODO 6: add gravity/Coriolis compensation d.qfrc_bias[self.dof]
        tau += d.qfrc_bias[dof]
        return tau