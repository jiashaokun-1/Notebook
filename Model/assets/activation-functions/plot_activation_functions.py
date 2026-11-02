"""Reproduce the activation figures using NumPy, SciPy and Matplotlib."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "activation-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.special import expit, ndtr, softmax


OUTPUT_DIR = Path(__file__).resolve().parent
COLORS = ["#2563eb", "#0891b2", "#8b5cf6", "#d97706", "#db2777", "#059669"]


def configure_style():
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = [family, "DejaVu Sans"]
            break
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1",
        "text.color": "#172033",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#e2e8f0",
        "grid.alpha": 0.7,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
    })


def prepare_axis(ax, xlabel="输入 x", ylabel="输出 f(x)"):
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.7)
    ax.axhline(0, color="#94a3b8", linewidth=0.9)
    ax.axvline(0, color="#94a3b8", linewidth=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-4, 4)
    ax.set_xticks([-4, -2, 0, 2, 4])


def save_figure(fig, stem):
    for suffix in ["png", "svg"]:
        target = OUTPUT_DIR / f"{stem}.{suffix}"
        fig.savefig(target, dpi=180)
        print(target.name)
    plt.close(fig)


def scalar_curves():
    x = np.linspace(-4, 4, 1601)
    specs = [
        ("ReLU", r"$f(x)=\max(0,x)$", np.maximum(x, 0), (-0.5, 4.4)),
        ("GELU（精确定义）", r"$f(x)=x\Phi(x)$", x * ndtr(x), (-0.5, 4.4)),
        ("SiLU / Swish-1", r"$f(x)=x\sigma(x)$", x * expit(x), (-0.5, 4.4)),
        ("Sigmoid", r"$f(x)=1/(1+e^{-x})$", expit(x), (-0.1, 1.1)),
        ("Tanh", r"$f(x)=\tanh(x)$", np.tanh(x), (-1.2, 1.2)),
        ("Softplus（补充）", r"$f(x)=\ln(1+e^x)$", np.logaddexp(0, x), (-0.5, 4.4)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("大模型常用激活函数：六种标量曲线", fontsize=22, y=0.972)
    fig.text(0.5, 0.925, "横轴范围相同；各图纵轴刻度不同。GELU 与 SiLU 可输出小幅负值。", ha="center")
    for ax, spec, color in zip(axes.flat, specs, COLORS):
        title, formula, y, limits = spec
        prepare_axis(ax)
        ax.plot(x, y, color=color, linewidth=2.7)
        ax.set_title(title + "\n" + formula, pad=13)
        ax.set_ylim(*limits)
    fig.tight_layout(rect=(0.015, 0.045, 0.995, 0.89), h_pad=2.6, w_pad=2.1)
    fig.text(0.5, 0.018, "按数学定义计算；Softplus 的 β=1。曲线不代表 kernel 耗时或模型精度。", ha="center", fontsize=10)
    save_figure(fig, "scalar-activation-curves")


def gated_curves():
    g = np.linspace(-4, 4, 1601)
    specs = [
        ("GLU", r"$y=u\,\sigma(g)$", expit(g)),
        ("ReGLU", r"$y=u\,\mathrm{ReLU}(g)$", np.maximum(g, 0)),
        ("GEGLU", r"$y=u\,\mathrm{GELU}(g)$", g * ndtr(g)),
        ("SwiGLU", r"$y=u\,\mathrm{SiLU}(g)$", g * expit(g)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("门控结构：固定内容分支 u 后的曲线切片", fontsize=21, y=0.974)
    fig.text(0.5, 0.925, "横轴是门控分支 g；每条线固定一个 u。未包含前后的线性投影。", ha="center")
    for ax, (title, formula, gate) in zip(axes.flat, specs):
        prepare_axis(ax, "门控分支 g", "输出 y")
        for u, color, style in zip([1, 2, -1], COLORS[:3], ["-", "--", "-."]):
            ax.plot(g, u * gate, label=f"u = {u}", color=color, linestyle=style, linewidth=2.5)
        ax.set_title(title + "\n" + formula, pad=12)
        ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
        ax.set_ylim((-1.3, 2.6) if title == "GLU" else (-4.8, 10))
    fig.tight_layout(rect=(0.015, 0.045, 0.99, 0.89), h_pad=2.6, w_pad=2.3)
    fig.text(0.5, 0.018, "这是二元函数 y(g,u) 的一维切片；SwiGLU 的 β=1。", ha="center", fontsize=10)
    save_figure(fig, "gated-activation-curves")


def softmax_curves():
    x = np.linspace(-4, 4, 1601)
    logits = [np.column_stack([x, np.zeros_like(x)]), np.column_stack([x, np.zeros_like(x), np.ones_like(x)])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.suptitle("Softmax：固定其余 logits 后的概率曲线", fontsize=21, y=0.962)
    fig.text(0.5, 0.875, "仅第一个 logit 随 x 改变；每个 x 处，同一图内的概率之和都为 1。", ha="center")
    for ax, z, title in zip(axes, logits, [r"$z=[x,0]$", r"$z=[x,0,1]$"]):
        probabilities = softmax(z, axis=1)
        prepare_axis(ax, "第一个 logit：x", "概率 p")
        for i in range(z.shape[1]):
            ax.plot(x, probabilities[:, i], label=f"p{i + 1}", color=COLORS[i], linewidth=2.6)
        ax.set_title(title, pad=13)
        ax.set_ylim(-0.05, 1.12)
        ax.legend(frameon=False, loc="upper left", ncol=3)
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.83), w_pad=3)
    fig.text(0.5, 0.025, "左图 p1=Sigmoid(x)；右图分母还包含 e¹，不能把每个 logit 独立当作 Sigmoid。", ha="center", fontsize=10)
    save_figure(fig, "softmax-curves")


def verify_values():
    x = np.array([-2., -1., 0., 1., 2.])
    values = {
        "ReLU": np.maximum(x, 0),
        "GELU": x * ndtr(x),
        "SiLU": x * expit(x),
        "Sigmoid": expit(x),
        "Tanh": np.tanh(x),
        "Softplus": np.logaddexp(0, x),
    }
    for name, y in values.items():
        assert np.all(np.isfinite(y)), name
        print(name, " ".join(f"{v:.4f}" for v in y))
    np.testing.assert_allclose(softmax([1, 2, 3]).sum(), 1.0)
    np.testing.assert_allclose(softmax(np.column_stack([x, np.zeros_like(x)]), axis=1)[:, 0], expit(x))
    np.testing.assert_allclose(values["GELU"][2], 0.0)
    np.testing.assert_allclose(values["Softplus"][2], np.log(2))
    print("Softmax([1,2,3]):", softmax([1., 2., 3.]).round(4).tolist())


if __name__ == "__main__":
    configure_style()
    verify_values()
    scalar_curves()
    gated_curves()
    softmax_curves()
