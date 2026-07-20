#!/usr/bin/env python3
"""Generate the concise Chinese Markdown and two-page PDF experiment report."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/experiment-final"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
OBJECTS = ("jar", "cellphone", "usb_stick")
LABELS = {"jar": "Jar", "cellphone": "Cellphone", "usb_stick": "USB stick"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fmt_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def object_stats(rows: list[dict], method: str, obj: str) -> tuple[float, float, int]:
    selected = [row for row in rows if row["method"] == method and row["object"] == obj]
    rates = np.asarray([float(row["success_rate"]) for row in selected])
    return float(rates.mean()), float(rates.std(ddof=1)), sum(int(row["success_count"]) for row in selected)


def find_render(obj: str, case: str) -> Path:
    candidates = sorted((OUT / "renders" / obj).glob(f"*/{case}/contact_sheet.png"))
    if not candidates:
        raise FileNotFoundError(f"No {case} contact sheet for {obj}")
    return candidates[0]


def build_markdown(best: list[dict], aggregate: list[dict]) -> str:
    aggregate_map = {(row["method"], row["object"]): row for row in aggregate}
    baseline_rate = float(aggregate_map[("baseline", "aggregate")]["mean_success_rate"])
    wrist4_rate = float(aggregate_map[("wrist4", "aggregate")]["mean_success_rate"])
    final_method = "wrist4" if wrist4_rate > baseline_rate else "baseline"
    final_cn = "wrist-weighted（wrist=4）" if final_method == "wrist4" else "baseline（wrist=2）"
    outcome = "带来正向提升" if wrist4_rate > baseline_rate else "未提高 aggregate 成功率，按预案作为负结果保留"
    phone_max = max(int(row["success_count"]) for row in best if row["object"] == "cellphone")
    usb_baseline = object_stats(best, "baseline", "usb_stick")[0]
    usb_wrist4 = object_stats(best, "wrist4", "usb_stick")[0]
    target = "达到" if max(baseline_rate, wrist4_rate) > 11 / 60 else "未超过"

    table = [
        "| 物体 | Baseline（3 seeds） | Wrist=4（3 seeds） |",
        "|---|---:|---:|",
    ]
    for obj in OBJECTS:
        b_mean, b_std, b_success = object_stats(best, "baseline", obj)
        w_mean, w_std, w_success = object_stats(best, "wrist4", obj)
        table.append(
            f"| {LABELS[obj]} | {b_success}/60，{b_mean*100:.2f}% ± {b_std*100:.2f}% | "
            f"{w_success}/60，{w_mean*100:.2f}% ± {w_std*100:.2f}% |"
        )
    b_agg = aggregate_map[("baseline", "aggregate")]
    w_agg = aggregate_map[("wrist4", "aggregate")]
    table.append(
        f"| Aggregate | {b_agg['successes']}/{b_agg['trajectories']}，{baseline_rate*100:.2f}% | "
        f"{w_agg['successes']}/{w_agg['trajectories']}，{wrist4_rate*100:.2f}% |"
    )

    jar_success = find_render("jar", "success").relative_to(OUT)
    phone_failure = find_render("cellphone", "failure").relative_to(OUT)
    return f"""# 灵巧手三物体抓取实验报告

## 1. 执行路线与公平协议

本实验使用 GraspM3 的 Jar、Cellphone、USB stick，各自训练单物体策略。数据按完整 sequence 固定 80/20 划分（split seed=0），训练 seed 为 0/1/2；batch size 256，学习率 `2e-4`，最多 200 epoch、validation patience 30，以最低 validation loss checkpoint 评估。Baseline 使用官方兼容 MLP 与 `2×wrist + orientation + finger + L1`；唯一改进是把 wrist 权重由 2 改为 4。

RTX 5070 Ti 的 compute capability 为 sm_120。官方环境固定的 PyTorch 1.12.1+cu113 只包含至 sm_86 的 CUDA kernel，因此策略张量不能在这张卡上执行；这不是“整张 GPU 不可用”。本实验采用兼容分层路线：训练由 PyTorch 2.7.1+cu128 在 RTX 5070 Ti 上完成；checkpoint 经旧 PyTorch 严格加载；物理评估使用 GPU PhysX（`sim_device=cuda:0`、CPU pipeline），策略推理放在 CPU。

## 2. 最终结果

{chr(10).join(table)}

三 seed 误差为样本标准差。wrist=4 相对 baseline {outcome}，最终策略选择 **{final_cn}**。公平实验的最佳 aggregate {target}前期联合模型参考值 11/60（18.33%）；二者训练协议不同，前期数字仅作参考。Cellphone 单次 best run 最高为 {phone_max}/20，说明 wrist 加权缓解了前期 wrist/姿态泛化瓶颈，但失败回放表明方向敏感问题仍未消失。代价是 USB 从 {usb_baseline*100:.2f}% 降到 {usb_wrist4*100:.2f}%，成为最终策略的主要短板。

![三物体成功率](figures/three_object_success_bars.png)

## 3. 曲线与真实回放证据

![训练与验证 BC loss](figures/bc_loss_curves.png)

![Checkpoint success 曲线](figures/checkpoint_success_curves.png)

![Checkpoint normalized lift 曲线](figures/checkpoint_normalized_lift_curves.png)

“checkpoint evaluation normalized lift score”定义为 `clip(max_lift_m / 0.30, 0, 1)`，用于补充闭环成功率，**不是 BC training reward**。每个成功率数字均可追溯至 `results/trajectory_metrics.csv` 和原始 `rollout_metrics.json` 汇总。

### 成功案例：Jar

![Jar 成功四帧]({jar_success.as_posix()})

### 失败案例：Cellphone

![Cellphone 失败四帧]({phone_failure.as_posix()})

所有案例均由真实 rollout 的手部 28 维状态、物体 7 维位姿和官方几何模型重建，使用相同相机、坐标范围、帧率与颜色；`renders/` 同时提供 GIF、MP4、四帧 PNG 和 metrics JSON。

## 4. 结论边界与下一步

本实验只验证了固定三物体、固定 20 条 raw trajectory 和两个 wrist 权重，不能外推到未见物体或真实机器人。下一步优先对 Cellphone 做 wrist 姿态分层诊断，同时检查 USB 对 wrist 权重的负迁移，并在不破坏公平协议的前提下扩大 sequence 与独立测试对象；本阶段不继续无边界调参。
"""


def add_text(ax, x: float, y: float, text: str, size: float = 9, bold: bool = False,
             width: int = 66, line_height: float = 0.043) -> float:
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=True) or [""])
    prop = FONT_BOLD if bold else FONT
    for line in lines:
        ax.text(x, y, line, fontproperties=prop, fontsize=size, va="top", transform=ax.transAxes)
        y -= line_height
    return y


def make_pdf(best: list[dict], aggregate: list[dict], pdf_path: Path) -> None:
    aggregate_map = {(row["method"], row["object"]): row for row in aggregate}
    b = aggregate_map[("baseline", "aggregate")]
    w = aggregate_map[("wrist4", "aggregate")]
    final_method = "wrist4" if float(w["mean_success_rate"]) > float(b["mean_success_rate"]) else "baseline"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes((0.07, 0.69, 0.86, 0.25)); ax.axis("off")
        y = add_text(ax, 0, 1.0, "灵巧手三物体抓取实验报告", 18, True, width=40, line_height=0.08)
        y = add_text(ax, 0, y, "公平协议", 12, True, width=40, line_height=0.06)
        y = add_text(ax, 0, y,
            "Jar、Cellphone、USB stick 分别训练；完整 sequence 固定 80/20 划分（split seed=0），训练 seed=0/1/2。batch size=256，learning rate=2e-4，最多 200 epoch，patience=30，最低 validation loss 选 checkpoint。Baseline 为 2×wrist + orientation + finger + L1；唯一改进是 wrist 权重改为 4。",
            size=9.2, width=72)
        y = add_text(ax, 0, y, "RTX 5070 Ti 兼容执行路线", 12, True, width=40, line_height=0.06)
        add_text(ax, 0, y,
            "RTX 5070 Ti 是 sm_120，官方 PyTorch 1.12.1+cu113 仅含至 sm_86 kernel，因此旧策略张量不能直接上 GPU。训练使用 PyTorch 2.7.1+cu128 + GPU；旧环境严格加载 checkpoint；回放由 GPU PhysX 执行，策略推理使用 CPU。GPU 并非完全不可用。",
            size=9.2, width=72)
        table_rows = []
        for obj in OBJECTS:
            bm, bs, bc = object_stats(best, "baseline", obj)
            wm, ws, wc = object_stats(best, "wrist4", obj)
            table_rows.append([
                LABELS[obj], f"{bc}/60，{bm*100:.1f}% ± {bs*100:.1f}%",
                f"{wc}/60，{wm*100:.1f}% ± {ws*100:.1f}%",
            ])
        table_rows.append([
            "Aggregate", f"{b['successes']}/{b['trajectories']}，{float(b['mean_success_rate'])*100:.1f}%",
            f"{w['successes']}/{w['trajectories']}，{float(w['mean_success_rate'])*100:.1f}%",
        ])
        table_ax = fig.add_axes((0.08, 0.51, 0.84, 0.16)); table_ax.axis("off")
        table = table_ax.table(
            cellText=table_rows, colLabels=["物体", "Baseline（3 seeds）", "Wrist=4（3 seeds）"],
            cellLoc="center", loc="center", colWidths=[0.18, 0.41, 0.41],
        )
        table.scale(1, 1.5)
        for (row, _), cell in table.get_celld().items():
            cell.get_text().set_fontproperties(FONT_BOLD if row == 0 else FONT)
            cell.get_text().set_fontsize(8.5)
            if row == 0:
                cell.set_facecolor("#e8eef6")
        image = plt.imread(OUT / "figures/three_object_success_bars.png")
        image_ax = fig.add_axes((0.08, 0.09, 0.84, 0.39)); image_ax.imshow(image); image_ax.axis("off")
        fig.text(0.5, 0.055,
                 f"Aggregate：baseline {b['successes']}/{b['trajectories']}；wrist=4 {w['successes']}/{w['trajectories']}；最终选择 {final_method}。",
                 ha="center", fontproperties=FONT, fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.965, "曲线、回放证据与结论", fontproperties=FONT_BOLD, fontsize=16, va="top")
        for position, filename in [
            ((0.06, 0.73, 0.88, 0.20), "bc_loss_curves.png"),
            ((0.06, 0.54, 0.88, 0.19), "checkpoint_success_curves.png"),
            ((0.06, 0.35, 0.88, 0.19), "checkpoint_normalized_lift_curves.png"),
        ]:
            ax_img = fig.add_axes(position); ax_img.imshow(plt.imread(OUT / "figures" / filename)); ax_img.axis("off")
        success_ax = fig.add_axes((0.06, 0.15, 0.43, 0.18))
        success_ax.imshow(plt.imread(find_render("jar", "success"))); success_ax.axis("off")
        success_ax.set_title("Jar 成功案例", fontproperties=FONT, fontsize=9)
        failure_ax = fig.add_axes((0.51, 0.15, 0.43, 0.18))
        failure_ax.imshow(plt.imread(find_render("cellphone", "failure"))); failure_ax.axis("off")
        failure_ax.set_title("Cellphone 失败案例", fontproperties=FONT, fontsize=9)
        note_ax = fig.add_axes((0.07, 0.025, 0.86, 0.10)); note_ax.axis("off")
        add_text(note_ax, 0, 1,
            "normalized lift = clip(max_lift_m / 0.30, 0, 1)，名称为 checkpoint evaluation normalized lift score，不是 BC training reward。所有结果可追溯到逐轨迹 JSON/CSV。wrist=4 缓解了 Cellphone 的历史 wrist/姿态瓶颈，但造成 USB 负迁移；本结论仅适用于固定三物体和固定 20 条 raw trajectory。下一步做 Cellphone 姿态分层和 USB 负迁移诊断，不继续无边界调参。",
            size=8.0, width=96, line_height=0.24)
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    best = read_csv(OUT / "results/best_rollout_summary.csv")
    aggregate = read_csv(OUT / "results/aggregate_summary.csv")
    markdown = build_markdown(best, aggregate)
    md_path = OUT / "灵巧手抓取实验报告.md"
    md_path.write_text(markdown, encoding="utf-8")
    pdf_path = OUT / "灵巧手抓取实验报告.pdf"
    make_pdf(best, aggregate, pdf_path)
    print(md_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
