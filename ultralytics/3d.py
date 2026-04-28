import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 恢复默认英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

def save_3d_surface(img, title, save_path, stride=4, z_label="Gradient Magnitude"):
    h, w = img.shape
    x = np.arange(0, w, stride)
    y = np.arange(0, h, stride)
    X, Y = np.meshgrid(x, y)
    Z = img[::stride, ::stride]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘图
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=False, alpha=0.9)
    
    # 全英文标注
    ax.set_title(title, fontsize=20, pad=20, fontweight='bold')
    ax.set_zlim(0, 255)
    ax.set_xlabel("Width (px)", fontsize=12, labelpad=10)
    ax.set_ylabel("Height (px)", fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.view_init(elev=55, azim=-60)
    
    cbar = plt.colorbar(surf, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label(z_label, fontsize=12)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"已保存 3D 图: {save_path}")

def save_polar_histogram(grad_x, grad_y, title, save_path):
    """生成极坐标直方图，内容全英文"""
    mag, ang = cv2.cartToPolar(grad_x, grad_y)
    mask = mag > 20
    valid_ang = ang[mask]
    
    if len(valid_ang) == 0:
        return

    bins_count = 36
    hist, bins = np.histogram(valid_ang, bins=bins_count, range=(0, 2*np.pi))
    width = 2 * np.pi / bins_count

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    bars = ax.bar(bins[:-1], hist, width=width, bottom=0.0)
    
    max_h = max(hist)
    for r, bar in zip(hist, bars):
        bar.set_facecolor(plt.cm.viridis(r / max_h))
        bar.set_alpha(0.85)
        bar.set_edgecolor('white')
        bar.set_linewidth(0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yticklabels([])
    
    plt.title(title, fontsize=18, pad=20, fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"已保存极坐标图: {save_path}")

def compute_visuals_v2(image_path, output_root):
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None: return

    filename = os.path.splitext(os.path.basename(image_path))[0]
    # 文件夹名保留中文，方便用户看
    save_dir = os.path.join(output_root, filename + "_3D极坐标分析")
    os.makedirs(save_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 原始梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 2. LCGR 增强
    k_size = 15
    grad_local_avg = cv2.boxFilter(grad_mag, -1, (k_size, k_size))
    grad_lcgr = np.zeros_like(grad_mag)
    noise_gate = 15.0
    valid_mask = grad_mag > noise_gate
    grad_lcgr[valid_mask] = grad_mag[valid_mask] / (grad_local_avg[valid_mask] + 1e-5)
    grad_lcgr = np.clip(grad_lcgr, 0, 3.0)
    grad_lcgr_norm = cv2.normalize(grad_lcgr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- 文件名中文，图内英文 ---
    save_3d_surface(grad_norm, "Raw Gradient Landscape", 
                   os.path.join(save_dir, "1_3d_原始梯度.jpg"), z_label="Gradient Magnitude")
    save_3d_surface(grad_lcgr_norm, "LCGR Enhanced Landscape", 
                   os.path.join(save_dir, "2_3d_LCGR增强.jpg"), z_label="Enhanced Magnitude")

    h, w = gray.shape
    center_roi_x = sobelx[h//2-50:h//2+50, w//2-50:w//2+50]
    center_roi_y = sobely[h//2-50:h//2+50, w//2-50:w//2+50]
    save_polar_histogram(center_roi_x, center_roi_y, "Gradient Orientation Distribution", 
                        os.path.join(save_dir, "3_极坐标_方向分布.jpg"))

def main():
    input_dir = r"d:\latex work\bigdata\bigdata\1示例"
    output_root = os.path.join(input_dir, "可视化结果_论文终稿")
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths = [p for p in image_paths if "vis_" not in os.path.basename(p)]
    
    for img_path in image_paths:
        compute_visuals_v2(img_path, output_root)

if __name__ == "__main__":
    main()
