# -*- coding: utf-8 -*-
"""
演示 ReLU 神经元在训练过程中失活的比例。

失活定义：在固定的完整训练集上，某个 ReLU 单元的输出始终为 0。
脚本会运行多个随机种子，并输出一张 PNG 图和一份 CSV 数据表。
"""

import os

# 当前 Anaconda 环境同时加载 Intel MKL 与 PyTorch 自带的 OpenMP 库时，
# 可能触发重复初始化警告。该设置只用于避免本机环境冲突。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 80
HIDDEN_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 0.12
NOISE = 0.13


class TwoLayerReLU(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

        # 略微负偏置能让较高学习率下的死亡 ReLU 现象更清楚，
        # 但不改变正常的前向传播与反向传播过程。
        nn.init.constant_(self.fc1.bias, -0.1)
        nn.init.constant_(self.fc2.bias, -0.1)

    def hidden_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = torch.relu(self.fc1(x))
        a2 = torch.relu(self.fc2(a1))
        return a1, a2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1, a2 = self.hidden_activations(x)
        return self.fc3(a2)


def make_two_moons(
    samples_per_class: int = 700, noise: float = NOISE, seed: int = 123
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成无需额外数据包的双月分类数据。"""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, samples_per_class)

    outer = np.column_stack((np.cos(theta), np.sin(theta)))
    inner = np.column_stack((1 - np.cos(theta), 0.5 - np.sin(theta)))
    x = np.vstack((outer, inner))
    y = np.concatenate(
        (
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        )
    )

    x += rng.normal(0, noise, size=x.shape)
    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def dead_fraction(model: TwoLayerReLU, x: torch.Tensor) -> tuple[float, float]:
    """统计每一层中，对全部输入都输出 0 的隐藏单元比例。"""
    model.eval()
    with torch.no_grad():
        a1, a2 = model.hidden_activations(x)
        dead_mask_1 = (a1 <= 1e-8).all(dim=0)
        dead_mask_2 = (a2 <= 1e-8).all(dim=0)
        return dead_mask_1.float().mean().item(), dead_mask_2.float().mean().item()


def train_one_seed(
    seed: int, x: torch.Tensor, y: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )

    model = TwoLayerReLU()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0
    )
    criterion = nn.CrossEntropyLoss()

    layer1_curve: list[float] = []
    layer2_curve: list[float] = []

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        layer1, layer2 = dead_fraction(model, x)
        layer1_curve.append(layer1)
        layer2_curve.append(layer2)

    return (
        np.asarray(layer1_curve),
        np.asarray(layer2_curve),
        np.asarray([(a + b) / 2 for a, b in zip(layer1_curve, layer2_curve)]),
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    x, y = make_two_moons()

    all_layer1: list[np.ndarray] = []
    all_layer2: list[np.ndarray] = []
    all_total: list[np.ndarray] = []

    for seed in SEEDS:
        layer1, layer2, total = train_one_seed(seed, x, y)
        all_layer1.append(layer1)
        all_layer2.append(layer2)
        all_total.append(total)
        print(
            f"seed={seed}: final dead fraction "
            f"layer1={layer1[-1]:.1%}, layer2={layer2[-1]:.1%}, "
            f"total={total[-1]:.1%}"
        )

    layer1_matrix = np.vstack(all_layer1)
    layer2_matrix = np.vstack(all_layer2)
    total_matrix = np.vstack(all_total)
    epochs = np.arange(1, EPOCHS + 1)

    csv_path = output_dir / "dead_relu_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("seed,epoch,layer1_dead_fraction,layer2_dead_fraction,total_dead_fraction\n")
        for row_index, seed in enumerate(SEEDS):
            for epoch_index in range(EPOCHS):
                f.write(
                    f"{seed},{epoch_index + 1},"
                    f"{layer1_matrix[row_index, epoch_index]:.8f},"
                    f"{layer2_matrix[row_index, epoch_index]:.8f},"
                    f"{total_matrix[row_index, epoch_index]:.8f}\n"
                )

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )

    mean_total = total_matrix.mean(axis=0)
    min_total = total_matrix.min(axis=0)
    max_total = total_matrix.max(axis=0)
    mean_layer1 = layer1_matrix.mean(axis=0)
    mean_layer2 = layer2_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax.fill_between(epochs, min_total, max_total, color="#B9D9F3", alpha=0.45, label="总计范围")
    ax.plot(epochs, mean_total, color="#C5422D", linewidth=3.0, label="总体平均")
    ax.plot(
        epochs,
        mean_layer1,
        color="#1F4E79",
        linewidth=2.0,
        linestyle="--",
        label="第 1 个隐藏层",
    )
    ax.plot(
        epochs,
        mean_layer2,
        color="#2E8B57",
        linewidth=2.0,
        linestyle=":",
        label="第 2 个隐藏层",
    )

    final_mean = mean_total[-1]
    ax.annotate(
        f"训练结束时平均 {final_mean:.1%}",
        xy=(epochs[-1], final_mean),
        xytext=(-145, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        fontsize=10,
        color="#333333",
    )

    ax.set_title("训练中 ReLU 失活神经元比例")
    ax.set_xlabel("训练轮次 Epoch")
    ax.set_ylabel("失活神经元比例")
    ax.set_ylim(-0.02, max(0.45, max_total.max() * 1.18))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()

    png_path = output_dir / "dead_relu_curve.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"已保存图片: {png_path}")
    print(f"已保存数据: {csv_path}")


if __name__ == "__main__":
    main()
