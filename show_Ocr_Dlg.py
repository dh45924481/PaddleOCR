import cv2
import torch
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import datetime
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import sys

# 添加拖拽功能的支持
USE_DND = True
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    USE_DND = False
    print(
        "提示：未找到 tkinterdnd2 模块。如需拖拽功能，请使用 'pip install tkinterdnd2' 安装。"
    )

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 如果是macOS系统，禁用不必要的IMK警告
if sys.platform == "darwin":
    os.environ["TK_SILENCE_DEPRECATION"] = "1"

# 实现文件夹图片浏览+识别检测， 拷贝， 拖拽图片，

# 全局变量
current_folder = ""
image_files = []
current_index = -1

device = "cuda"  # 您可以根据需要更改为 'cuda'，如果您的GPU支持
print(f"Using device: {device}")

model = PaddleOCR(
    det_model_dir="./inference/ch_PP-OCRv4_det_server_infer/",
    # rec_model_dir='./inference/ch_PP-OCRv4_rec_infer/',
    rec_model_dir="./inference/ch_PP-OCRv4_rec_server_infer/",  # ok
    use_angle_cls=True,
    lang="ch",
    use_gpu=False,
    det_limit_side_len=1280,
)

# model = PaddleOCR(
#     det_model_dir='./inference/ch_PP-OCRv4_det_server_infer/',
#     rec_model_dir='./inference/rec_repsvtr_infer/',
#     # rec_model_dir='./inference/ch_PP-OCRv4_rec_server_infer/',  # ok
#     cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/',  #
#     # rec_char_dict_path = './shipname_dict.txt',
#     use_angle_cls=True,
#     lang='ch',
#     max_side_len=1000,
#     use_gpu=False,
#     show_log=True,
#     # 修改检测模型参数
#     det_db_thresh=0.2,          # 进一步降低检测阈值
#     det_db_box_thresh=0.5,      # 降低框阈值
#     det_db_unclip_ratio=2.0,    # 增加文本框扩张比例
#     # 识别配置
#     rec_batch_num=1,
#     # rec_algorithm='SVTR_LCNet',
#     # 添加额外的参数
#     drop_score=0.5,             # 降低文本识别置信度阈值
#     min_subbox_size=8,          # 最小检测文本大小
#     max_batch_size=10,          # 批处理大小
#     use_dilation=True,          # 使用膨胀
#     det_limit_side_len=1280     # 检测模型的最大边长
# )


def ShowRlt(ocr, img_path, det=True, text_output=None):
    if text_output:
        text_output.delete(1.0, tk.END)  # 清空文本框

    result = ocr.ocr(img_path, det=det, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            if text_output:
                text_output.insert(tk.END, str(line) + "\n")
            print(line)


def detect_and_track_ship(image, text_output=None):
    try:
        # 检查输入是文件路径还是图像对象
        if isinstance(image, str):
            frame = cv2.imread(image)
            image_path = image
        else:
            if isinstance(image, np.ndarray):
                frame = image
            else:
                frame = np.array(image)

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image_path = "剪贴板图像"

        frame = cv2.resize(frame, (1024, 768))
        result = model.ocr(frame, det=True, cls=True)

        # 清空文本框
        if text_output:
            text_output.delete(1.0, tk.END)

        # 简单直接地输出识别结果
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                output_text = str(line) + "\n"
                # 同时输出到控制台和文本框
                print(output_text)
                if text_output:
                    text_output.insert(tk.END, output_text)

        return frame, image_path
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None


# model = YOLO(r"best.pt").to(device)

# def detect_and_track_ship(image):

#     try:
#         # 检查输入是文件路径还是图像对象
#         if isinstance(image, str):
#             frame = cv2.imread(image)
#             image_path = image
#         else:
#             frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             image_path = '剪贴板图像'

#         frame = cv2.resize(frame, (1024, 768))
#         # results = model.track(frame, persist=True, verbose=False)
#         results = model.predict(frame, conf=0.5, device=device)
#         detections = []

#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#                 width = x2 - x1
#                 height = y2 - y1
#                 confidence = float(box.conf.cpu().numpy())
#                 class_id = int(box.cls.cpu().numpy())
#                 if class_id != 118:  # 过滤特定的类别ID
#                     detections.append([x1, y1, width, height, confidence])
#                     label = f"ID:{box.id[0]  if box.id is not None else 'N'}"
#                     print(label)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 150, 60), 2)
#                     cv2.putText(frame, label, (x2 - 80, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 240, 200), 1)

#         # 返回处理后的帧和图像路径
#         return frame, image_path
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return None, None


def select_image():
    image_path = filedialog.askopenfilename()
    if len(image_path) > 0:
        load_images_from_folder(image_path)
        process_and_show_image()


def process_and_show_image():
    if current_index >= 0 and current_index < len(image_files):
        image_path = os.path.join(current_folder, image_files[current_index])
        # 在这里传入text_output参数
        processed_frame, image_path = detect_and_track_ship(image_path, text_output)
        if processed_frame is not None:
            display_image(processed_frame, image_path)
    else:
        print("当前没有可处理的图片。")


def process_pasted_image():
    image = ImageGrab.grabclipboard()
    if image is not None:
        # 在这里传入text_output参数
        processed_frame, image_path = detect_and_track_ship(image, text_output)
        if processed_frame is not None:
            display_image(processed_frame, image_path)
            global current_folder, image_files, current_index
            current_folder = ""
            image_files = []
            current_index = -1
    else:
        print("剪贴板中没有可用的图像。")


def display_image(processed_frame, image_path):
    # 将OpenCV的BGR图像转换为PIL的RGB图像
    image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    # 根据需要调整图像大小以适应GUI窗口
    pil_image = pil_image.resize((800, 600), Image.LANCZOS)
    # 将PIL图像转换为ImageTk格式
    tk_image = ImageTk.PhotoImage(pil_image)
    # 更新标签中的图像
    image_label.config(image=tk_image)
    image_label.image = tk_image
    # 更新文件名文本框
    file_path_entry.config(state="normal")
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, image_path)
    file_path_entry.config(state="readonly")
    # 更新窗口标题为当前文件名
    root.title(f"识别效果展示 - {os.path.basename(image_path)}")


def load_images_from_folder(image_path):
    global current_folder, image_files, current_index
    current_folder = os.path.dirname(image_path)
    # 获取文件夹中所有支持的图片文件
    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    image_files = [
        f
        for f in os.listdir(current_folder)
        if f.lower().endswith(supported_extensions)
    ]
    image_files.sort()  # 按文件名排序
    # 获取当前图片的索引
    current_index = image_files.index(os.path.basename(image_path))
    print(f"已加载文件夹中的图片，共 {len(image_files)} 张，当前索引 {current_index}。")


def previous_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        process_and_show_image()
    else:
        print("已经是第一张图片。")


def next_image():
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        process_and_show_image()
    else:
        print("已经是最后一张图片。")


def process_dropped_file(image_path):
    if os.path.isfile(image_path):
        # 检查文件是否为图片
        if image_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            load_images_from_folder(image_path)
            process_and_show_image()
        else:
            print("不是图片文件：", image_path)
    else:
        print("不是文件：", image_path)


def drop(event):
    # 获取拖入的文件路径
    files = root.tk.splitlist(event.data)
    for f in files:
        process_dropped_file(f)


def on_paste(event):
    process_pasted_image()


if __name__ == "__main__":
    try:
        root = TkinterDnD.Tk() if USE_DND else tk.Tk()
    except Exception as e:
        root = tk.Tk()
        print(f"初始化拖拽功能时出错：{str(e)}，将使用基本模式。")

    try:
        root.title("识别效果展示")
        root.geometry("1000x800")

        # 创建主框架
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部框架
        top_frame = tk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_select = tk.Button(top_frame, text="导入图片", command=select_image)
        btn_select.pack(side=tk.LEFT, padx=5, pady=5)

        file_path_entry = tk.Entry(top_frame, width=80, state="readonly")
        file_path_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # 中间图片显示区域
        image_label = tk.Label(main_frame)
        image_label.pack(expand=True, fill=tk.BOTH)

        # 文本输出区域
        text_frame = tk.Frame(main_frame)
        text_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        text_scrollbar = tk.Scrollbar(text_frame)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_output = tk.Text(text_frame, height=10, yscrollcommand=text_scrollbar.set)
        text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.config(command=text_output.yview)

        # 底部按钮框架
        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = tk.Button(bottom_frame, text="上一张", command=previous_image)
        btn_prev.pack(side=tk.LEFT, padx=10, pady=5)

        btn_next = tk.Button(bottom_frame, text="下一张", command=next_image)
        btn_next.pack(side=tk.RIGHT, padx=10, pady=5)

        # 设置拖拽功能
        if USE_DND:
            try:
                image_label.drop_target_register(DND_FILES)
                image_label.dnd_bind("<<Drop>>", drop)
            except Exception as e:
                print(f"设置拖拽功能时出错：{str(e)}")

        # 绑定快捷键
        root.bind("<Control-v>", on_paste)
        root.bind("<Control-V>", on_paste)

        root.mainloop()
    except Exception as e:
        messagebox.showerror("错误", f"程序运行出错：{str(e)}")
