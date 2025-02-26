from paddleocr import PaddleOCR, draw_ocr
import os
import time

os.chdir(os.path.dirname(__file__))


def ShowRlt(ocr, img_path, det=True):
    start_time = time.time()  # 开始时间

    result = ocr.ocr(img_path, det=det, cls=True)

    end_time = time.time()  # 结束时间
    process_time = end_time - start_time  # 计算处理时间

    print(f"\n识别耗时: {process_time:.3f}秒")  # 打印处理时间
    print("\n识别结果:")
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)


ocr4model = PaddleOCR(
    det_model_dir="./inference/ch_PP-OCRv4_det_server_infer/",
    rec_model_dir="./inference/ch_PP-OCRv4_rec_server_infer/",  # ok
    use_angle_cls=False,
    lang="ch",
    det_limit_side_len=1280,
    use_gpu=True,
)

ShowRlt(ocr4model, "images/1.png")
