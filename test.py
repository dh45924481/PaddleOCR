from paddleocr import PaddleOCR, draw_ocr
def ShowRlt(ocr, img_path, det=True):
    """
    显示OCR识别结果。

    调用OCR引擎对指定图片进行文字识别，并打印识别结果。该函数不仅执行文字的检测和识别任务，
    还负责将识别出的文字信息输出到控制台。输出包括文字区域的坐标和识别出的文本内容。

    参数:
    ocr: OCR引擎实例，用于执行文字识别。
    img_path: 图片文件路径，指定需要识别的图片。
    det: 布尔值，指示是否执行文字检测。默认为True，如果设置为False，则直接进行文字识别，而不先进行检测。

    返回:
    该函数没有返回值，但会将识别结果直接打印到控制台。
    """
    # 执行OCR识别，包含检测、识别和分类步骤
    result = ocr.ocr(img_path, det=det, cls=True)

    # 遍历识别结果，每个元素包含一个检测到的文字区域及其识别结果

    """
    使用`for idx in range(len(result))`遍历`result`列表。`result`是由OCR引擎返回的识别结果，其中每个元素代表一个检测到的文字区域。
    通过`res = result[idx]`获取当前区域的文字识别结果。`res`是一个列表，包含该区域内的多行文字信息。
    使用`for line in res`遍历当前区域的每一行文字。
    通过`print(line)`将每行文字信息打印到控制台。`line`是一个包含文字区域坐标和识别出的文本内容的字典或列表。
    """

    for idx in range(len(result)):
        res = result[idx]
        # 遍历当前区域的文字行识别结果，并打印
        for line in res:
            print(line)


ocr4 = PaddleOCR(
    det_model_dir='./inference/ch_PP-OCRv4_det_server_infer/',
    rec_model_dir='./inference/ch_ppocr_server_v2.0_rec_infer/',
    # rec_model_dir='./inference/ch_PP-OCRv4_rec_server_infer/',
    use_angle_cls=True,
    lang='ch',
    # use_gpu=False
)

# ShowRlt(ocr4,r'D:\test2024\PaddleOCR\train_data\rec\train\100007_禹顺189_3_crop_2.jpg')

ShowRlt(ocr4,'/Users/dh/Desktop/ID218_2024-12-12-12-29-02.550_112_666.jpg')
