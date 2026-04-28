import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 尝试导入 matplotlib 来寻找字体，这是最稳健的方法
try:
    import matplotlib.font_manager as fm
except ImportError:
    fm = None


# 1. 图片路径
IMAGE_PATH = "/root/ultralytics/ultralytics/cfg/RUOD/valid/images/001169_jpg.rf.5030f2197fce93df84998f8f8306b0a4.jpg"

# 2. 模型配置文件路径 (建议填 None，除非你有特殊需求)
MODEL_YAML = None 
# 3. 模型权重文件路径
MODEL_WEIGHTS = "/root/ultralytics/runs/detect/26sSHFI/weights/best.pt" 
# 4. 结果保存文件夹
OUTPUT_DIR = "/root/ultralytics/runs/detect/结果图"
# 5. 置信度阈值
CONF_THRES = 0.3

# 6. IoU 阈值 (判断预测框是否匹配上真实框)
IOU_THRES = 0.5

FONT_PATH = None

# 示例: "custom_name.jpg" 或 None
OUTPUT_FILENAME = "26sSHFIbest.jpg"
# ===================================================================

FILE = Path(__file__).resolve()
PARENT_DIR = FILE.parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 无法导入 'ultralytics' 包。")
    sys.exit(1)

def xywhn2xyxy(x, w, h):
    """Convert normalized [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2)  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2)  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2)  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2)  # bottom right y
    return y

def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    box1: [M, 4] (GT)
    box2: [N, 4] (Pred)
    Returns: [M, N] matrix
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def draw_overlay_text(image, text, text_color, position='top-left'):

    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 创建一个用于绘制半透明层的图层
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = 60 # 字体大小
    font = None
    
    # 1. 优先尝试配置的 FONT_PATH
    if FONT_PATH and os.path.exists(FONT_PATH):
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception as e:
            print(f"加载指定字体失败: {e}")

    # 2. 使用 matplotlib 自动寻找系统字体
    if font is None and fm is not None:
        try:
            sys_font_path = fm.findfont(fm.FontProperties(family='sans-serif', weight='bold'))
            if os.path.exists(sys_font_path):
                font = ImageFont.truetype(sys_font_path, font_size)
        except Exception as e:
            pass

    # 3. 硬编码回退
    if font is None:
        try:
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "arial.ttf"
            ]
            for path in font_paths:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, font_size)
                    break
        except:
            pass

    # 4. 默认字体
    if font is None:
        font = ImageFont.load_default()

    # 计算文字宽高
    try:
        # textbbox 返回 (left, top, right, bottom)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    # 位置设置
    padding = 10 
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2
    
    img_w, img_h = image.size
    
    if position == 'top-right':
        # 右上角
        bg_x1 = img_w - box_w
        bg_y1 = 0
        bg_x2 = img_w
        bg_y2 = box_h
    else:
        # 默认左上角
        bg_x1 = 0
        bg_y1 = 0
        bg_x2 = box_w
        bg_y2 = box_h

    # 绘制半透明黑色背景 (R, G, B, Alpha)
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 180))

    # 绘制文字 (居中)
    cx = (bg_x1 + bg_x2) / 2
    cy = (bg_y1 + bg_y2) / 2
    
    # 确保 text_color 是 RGBA 格式
    if len(text_color) == 3:
        text_color = text_color + (255,)
        
    try:
        # anchor='mm' 表示文字中心对齐
        draw.text((cx, cy), text, fill=text_color, font=font, anchor='mm')
    except TypeError:
        # 兼容旧版 Pillow
        # 手动计算居中位置 (假设 draw.text 默认以左上角为起点)
        x = cx - text_w / 2
        y = cy - text_h / 2
        draw.text((x, y), text, fill=text_color, font=font)
    
    # 合成图层
    return Image.alpha_composite(image, overlay).convert('RGB')

def find_label_path(image_path):
    # 方法1：直接简单替换
    candidate1 = image_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
    if os.path.exists(candidate1):
        return candidate1

    # 方法2：分割路径替换最后一个 images
    try:
        norm_path = os.path.normpath(image_path)
        parts = norm_path.split(os.sep)
        if 'images' in parts:
            rev_idx = parts[::-1].index('images')
            idx = len(parts) - 1 - rev_idx
            parts[idx] = 'labels'
            filename_no_ext = os.path.splitext(parts[-1])[0]
            parts[-1] = filename_no_ext + ".txt"
            candidate2 = os.sep.join(parts)
            if os.path.exists(candidate2):
                return candidate2
    except Exception:
        pass

    # 方法3：同级目录
    candidate3 = image_path.rsplit('.', 1)[0] + ".txt"
    if os.path.exists(candidate3):
        return candidate3
        
    return candidate1 

def main():
    print("========== 开始检测 ==========")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 找不到图片文件: {IMAGE_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        except:
            pass

    # 1. 加载模型
    print("正在加载模型...")
    try:
        if MODEL_YAML:
            model = YOLO(MODEL_YAML, task='detect')
            if MODEL_WEIGHTS:
                model.load(MODEL_WEIGHTS)
        else:
            model = YOLO(MODEL_WEIGHTS, task='detect')
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 寻找对应的标签文件
    label_path = find_label_path(IMAGE_PATH)
    print(f"推测标签路径: {label_path}")

    gt_boxes = []
    has_label = False
    
    # 读取图片获取尺寸
    try:
        img_cv = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), -1)
    except:
        img_cv = cv2.imread(IMAGE_PATH)
        
    if img_cv is None:
        print(f"无法读取图片: {IMAGE_PATH}")
        return
    img_h, img_w = img_cv.shape[:2]

    if os.path.exists(label_path):
        print(f"√ 成功找到标签文件")
        has_label = True
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    box_norm = np.array([parts[1], parts[2], parts[3], parts[4]])
                    box_xyxy = xywhn2xyxy(box_norm, img_w, img_h)
                    gt_boxes.append(box_xyxy)
        gt_boxes = np.array(gt_boxes)
        print(f"  共有 {len(gt_boxes)} 个真实目标 (GT)")
    else:
        print(f"× 未找到标签文件，将无法计算漏检和多检")

    # 3. 预测
    print(f"正在检测...")
    results = model.predict(source=IMAGE_PATH, conf=CONF_THRES, save=False)
    
    if not results:
        print("预测无结果")
        return

    result = results[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy() # [N, 4]
    print(f"  检测到 {len(pred_boxes)} 个预测目标")

    # 4. 计算漏检和多检
    missed_count = 0
    extra_count = 0
    
    if has_label:
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            # ious: [M, N] (M个GT, N个Pred)
            ious = box_iou(gt_boxes, pred_boxes)
            
            # --- 计算漏检 (Missed) ---
            max_ious_per_gt = ious.max(axis=1) # [M]
            missed_count = np.sum(max_ious_per_gt < IOU_THRES)
            
            # --- 计算多检 (Extra/False Positive) ---
            max_ious_per_pred = ious.max(axis=0) # [N]
            extra_count = np.sum(max_ious_per_pred < IOU_THRES)
            
        elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
            missed_count = len(gt_boxes)
            extra_count = 0
        elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
            missed_count = 0
            extra_count = len(pred_boxes)
        else:
            missed_count = 0
            extra_count = 0
            
    # 5. 绘图
    im_bgr = result.plot()
    im_rgb = im_bgr[..., ::-1]
    im_pil = Image.fromarray(im_rgb)

    # 6. 添加状态标签
    if has_label:
        # 左上角：漏检状态
        # 逻辑修改：统一显示 "N Fail" (N>=0)
        text_miss = f"{missed_count} Fail"
        color_miss = (255, 255, 255) # 白色
        
        im_pil = draw_overlay_text(im_pil, text_miss, color_miss, position='top-left')
        
        # 右上角：多检状态 (Extra)
        if extra_count > 0:
            text_extra = f"{extra_count} Extra"
            color_extra = (255, 255, 255) # 白色
        else:
            text_extra = "0 Extra"
            color_extra = (200, 200, 200) # 灰色
            
        im_pil = draw_overlay_text(im_pil, text_extra, color_extra, position='top-right')
        
    else:
        im_pil = draw_overlay_text(im_pil, "No Label", (200, 200, 200), position='top-left')

    # 7. 保存
    img_name = os.path.basename(IMAGE_PATH)
    
    if OUTPUT_FILENAME:
        save_name = OUTPUT_FILENAME
    else:
        save_name = f"result_{img_name}"
        
    save_path = os.path.join(OUTPUT_DIR, save_name)
    try:
        im_pil.save(save_path)
        print(f"========== 检测完成 ==========")
        print(f"结果已保存到: {save_path}")
        if has_label:
            print(f"统计: {missed_count} 个漏检 (Fail), {extra_count} 个多检 (Extra)")
    except Exception as e:
        print(f"保存失败: {e}")

if __name__ == "__main__":
    main()
