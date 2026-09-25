# NeuroForge 部署文档

## 1. 依赖锁定与复现

- `requirements.txt`：核心依赖（numpy / gymnasium / pyyaml / pytest），pin 死。
- `requirements-sota.txt`：SOTA 后端（stable-baselines3 / optuna / torch），pin 死。
- `requirements.lock.txt`：由 `pip freeze` 生成的完整锁定清单，干净环境可据此精确复现。
- `pyproject.toml`：声明式元数据 + 可选依赖组 `sota`，供 `pip install -e .[sota]` 使用。

## 2. 本地一键复现

```bash
git clone https://github.com/CJX0712/neuroforge.git
cd neuroforge
make install      # 建 .venv 并装核心 + SOTA 依赖
make demo         # 训练 CartPole 并评测
make test         # 运行全部单测
make benchmark    # 对标 DQN vs SB3(PPO)
```

说明：`make install` 会在仓库内创建 `.venv`（隔离、不污染全局）。若仅需离线兜底链路，可只装核心依赖：

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m neuroforge.examples.run_demo
```

## 3. Docker

```dockerfile
FROM python:3.13-slim
COPY . .
RUN pip install -r requirements.txt && pip install -r requirements-sota.txt && pip install -e .
CMD ["python", "-m", "neuroforge.examples.run_demo"]
```

构建运行：

```bash
docker build -t neuroforge .
docker run --rm neuroforge
```

## 4. 降级与可移植性

- 若目标机无法安装 `torch`（如受限架构），跳过 `requirements-sota.txt`，系统自动以纯 numpy DQN 运行（无 PPO/Optuna）。
- 全部随机种子由 `RLConfig.seed` 控制，训练可复现。
