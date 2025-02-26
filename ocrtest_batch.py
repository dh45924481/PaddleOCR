from paddleocr import PaddleOCR, draw_ocr
import os
import time
import asyncio
import cv2

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


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None

    # 调整图像大小
    img = cv2.resize(img, (960, 960))  # 调整为960x960的大小

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用二值化
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary


async def process_image(img_path, ocr):
    print(f"处理图片: {img_path}")
    img = preprocess_image(img_path)  # 预处理图像
    if img is None:
        return img_path

    # 使用 asyncio.to_thread 将同步操作转换为异步操作
    result = await asyncio.to_thread(ocr.ocr, img, det=True, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    return img_path


async def process_all_images(images):
    ocr = init_ocr()  # 创建一个OCR实例供所有任务共享
    tasks = []
    for img_path in images:
        task = asyncio.create_task(process_image(img_path, ocr))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    images = ["images/1.png", "images/2.png", "images/3.png", "images/4.png"]
    start_time = time.time()

    # 运行异步主函数
    asyncio.run(process_all_images(images))

    end_time = time.time()
    print(f"\n处理 {len(images)} 张图片总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每张图片处理时间: {(end_time - start_time)/len(images):.2f} 秒")
