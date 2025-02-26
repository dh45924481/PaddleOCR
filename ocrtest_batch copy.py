from paddleocr import PaddleOCR, draw_ocr
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

os.chdir(os.path.dirname(__file__))


def ShowRlt(ocr, img_path, det=True):
    result = ocr.ocr(img_path, det=det, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)


def init_ocr():
    return PaddleOCR(
        det_model_dir="./inference/ch_PP-OCRv4_det_server_infer/",
        # rec_model_dir='./inference/ch_PP-OCRv4_rec_infer/',
        rec_model_dir="./inference/ch_PP-OCRv4_rec_server_infer/",  # ok
        # cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/',  # 添加这行
        use_angle_cls=False,
        lang="ch",
        det_limit_side_len=1280,
        use_gpu=True,
    )


# ShowRlt(ocr4model, "images/1.png")


def process_image(img_path):
    ocr = init_ocr()  # 在每个进程中初始化新的OCR实例
    print(f"处理图片: {img_path}")
    result = ocr.ocr(img_path, det=True, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    return img_path


if __name__ == "__main__":
    images = ["images/1.png", "images/2.png", "images/3.png", "images/4.png"]
    start_time = time.time()  # 记录总开始时间

    # 获取CPU核心数，但最多使用4个进程
    num_processes = min(4, multiprocessing.cpu_count())
    print(f"\n使用进程数: {num_processes}")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_image, images))

    end_time = time.time()  # 记录总结束时间
    print(f"\n处理 {len(images)} 张图片总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每张图片处理时间: {(end_time - start_time)/len(images):.2f} 秒")
