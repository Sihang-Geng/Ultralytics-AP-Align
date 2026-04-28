import gc
import os
from typing import List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["axes.unicode_minus"] = False

VIS_VMAX = 200
VIS_GAMMA = 0.65
EDGE_COLOR = "#1B2A41"
EDGE_ALPHA = 0.88
EDGE_LW_DASH = 2.4
EDGE_LW_SOLID = 2.0
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def get_gradient(img_gray: np.ndarray) -> np.ndarray:
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def process_single_image(img_path: str, save_path: str) -> bool:
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read: {img_path}")
        return False

    target_size = 400
    img = cv2.resize(img, (target_size, target_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = get_gradient(gray)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    stride_grad = 4
    x_grad = np.arange(0, target_size, stride_grad)
    y_grad = np.arange(0, target_size, stride_grad)
    xg, yg = np.meshgrid(x_grad, y_grad)

    x_img = np.arange(0, target_size + 1, 1)
    y_img = np.arange(0, target_size + 1, 1)
    xi, yi = np.meshgrid(x_img, y_img)
    ceiling_z = 360

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    norm = mcolors.PowerNorm(gamma=VIS_GAMMA, vmin=0, vmax=VIS_VMAX)

    ax.plot_surface(
        xg,
        yg,
        grad[::stride_grad, ::stride_grad],
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=False,
        alpha=1.0,
        rstride=1,
        cstride=1,
        shade=False,
    )

    wall_z = np.full_like(xi, ceiling_z)
    ax.plot_surface(
        xi,
        yi,
        wall_z,
        facecolors=rgb_img,
        shade=False,
        linewidth=0,
        alpha=1.0,
        rstride=1,
        cstride=1,
        antialiased=False,
    )

    corners_x = [0, target_size, target_size, 0]
    corners_y = [0, 0, target_size, target_size]
    for cx, cy in zip(corners_x, corners_y):
        ax.plot(
            [cx, cx],
            [cy, cy],
            [0, ceiling_z],
            color=EDGE_COLOR,
            linestyle=(0, (2, 2)),
            linewidth=EDGE_LW_DASH,
            alpha=EDGE_ALPHA,
        )

    ax.plot([0, target_size, target_size, 0, 0], [0, 0, target_size, target_size, 0], [ceiling_z] * 5, color=EDGE_COLOR, linewidth=EDGE_LW_SOLID, alpha=EDGE_ALPHA)
    ax.plot([0, target_size, target_size, 0, 0], [0, 0, target_size, target_size, 0], [0] * 5, color=EDGE_COLOR, linewidth=EDGE_LW_SOLID, alpha=EDGE_ALPHA)

    ax.view_init(elev=26, azim=-58)
    ax.set_zlim(0, ceiling_z)
    ax.set_xlim(0, target_size)
    ax.set_ylim(0, target_size)
    ax.set_axis_off()
    ax.set_proj_type("persp")
    ax.set_box_aspect((1, 1, 0.88))

    plt.tight_layout(pad=0)
    fig.savefig(save_path, dpi=350, bbox_inches="tight", pad_inches=0.01, facecolor="white", transparent=False)
    plt.close(fig)
    gc.collect()
    return True


def find_numbered_file(input_dir: str, idx: int) -> Optional[str]:
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(input_dir, f"{idx}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def render_fixed_1_to_8(input_dir: str, single_output_dir: str) -> List[Tuple[int, str]]:
    os.makedirs(single_output_dir, exist_ok=True)
    rendered_items: List[Tuple[int, str]] = []
    for idx in range(1, 9):
        src = find_numbered_file(input_dir, idx)
        if src is None:
            print(f"[WARN] Missing numbered file: {idx}.[jpg|png|...]")
            continue
        out = os.path.join(single_output_dir, f"scene_{idx}.png")
        print(f"[Render] {os.path.basename(src)} -> {os.path.basename(out)}")
        if process_single_image(src, out):
            rendered_items.append((idx, out))
    return rendered_items


def assemble_2x4(rendered_items: List[Tuple[int, str]], output_path: str) -> None:
    path_map = {idx: path for idx, path in rendered_items}
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 10))
    labels = [str(i) for i in range(1, 9)]

    for i, ax in enumerate(axes.flatten()):
        idx = i + 1
        img_path = path_map.get(idx)
        if img_path is None or (not os.path.exists(img_path)):
            ax.axis("off")
            continue
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.5, -0.07, labels[i], transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")

    norm = mcolors.PowerNorm(gamma=VIS_GAMMA, vmin=0, vmax=VIS_VMAX)
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.36, 0.02, 0.28, 0.02])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 50, 100, 150, 200])
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10, wspace=0.03, hspace=0.14)
    fig.savefig(output_path, dpi=420, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "不同场景图", "筛选")
    single_output_dir = os.path.join(input_dir, "单图结果_1to8")
    output_dir = os.path.join(script_dir, "画图资料库")
    os.makedirs(output_dir, exist_ok=True)

    print("=== Step 1: render numbered scenes 1~8 ===")
    items = render_fixed_1_to_8(input_dir, single_output_dir)

    print("\n=== Step 2: assemble 2x4 figure ===")
    final_output_path = os.path.join(output_dir, "Fig_8_Scenes_Final.png")
    assemble_2x4(items, final_output_path)

    print("\n[Done]")
    print(f"- Output figure: {final_output_path}")
    print("- Workflow: rename approved candidate files to 5/6/7/8 and rerun this script.")
