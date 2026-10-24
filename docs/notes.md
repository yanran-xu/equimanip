# Notes

## Python

**`__call__`**：让对象可以像函数一样被调用。下面两行等价：

```python
tau = osc(target_pos, target_quat)
tau = osc.__call__(target_pos, target_quat)
```

## MuJoCo

**`self.qadr` 是地址，不是位置**
- 记录 7 个手臂关节的角度存在 `d.qpos` 的第几格，本场景为 `[0, 1, 2, 3, 4, 5, 6]`
- 整数，来自模型结构 `model`，仿真过程中不变
- 当前关节角 = `d.qpos[self.qadr]`

## OSC

**零空间（null space）**
- 自运动（self-motion）：关节在动，手不动。例如手按住桌面、转动手肘
- 所有这类运动方向的集合 = J 的零空间：`J · q̇ = 0`

**奇异位形附近为什么抖（pinv + rcond）**
1. rcond 是硬开关：σ 在阈值附近来回穿越（10 s 内切换 165 次），力跳变，来回推拉
2. 阈值上方力仍然很大：1/σ 可达 100+，力矩超出电机上限被截断 → bang-bang 振荡
3. 被丢弃的方向连阻尼也没了：`kd` 项也要乘 Λ，该方向振荡停不下来

## pytest

```bash
python -m pytest -q                  # 跑所有测试，简短输出
python -m pytest -v                  # 详细输出，显示每个测试名
python -m pytest tests/test_osc.py   # 只跑一个文件
python -m pytest -q -x               # 遇到第一个失败就停
python -m pytest -q -s               # 显示 print() 输出（默认隐藏）
```
