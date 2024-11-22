# 导入必要的库和模块
import os
import cv2
import numpy as np
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
import torch
from util.misc import to_device
from util.read_meter import MeterReader
from util.converter import keys, StringLabelConverter
from get_meter_area import Detector

# 解析命令行参数
option = BaseOptions()
args = option.initialize()

# 更新配置并打印
update_config(cfg, args)
print_config(cfg)

# 定义预测目录
predict_dir = 'demo/'

# 初始化模型并加载预训练权重
model = TextNet(is_training=False, backbone=cfg.net)
model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                          'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
print(model_path)
model.load_model(model_path)
model = model.to(cfg.device)
converter = StringLabelConverter(keys)

# 初始化检测器和表盘读数器
det = Detector()
detector = TextDetector_mask(model)
meter = MeterReader()
transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)

# 获取预测目录中的图像列表
image_list = os.listdir(predict_dir)

# 遍历图像列表进行处理
for index in image_list:
    print("**************", index)
    image = cv2.imread(predict_dir + index)

    # 显示原始图像
    cv2.imshow("det1", image)
    cv2.waitKey(0)

    # 检测表盘区域
    image, image_info, digital_list, meter_list = det.detect(image, index)

    # 如果没有检测到表盘则跳过
    if len(meter_list) == 0:
        print("no detected meter")
        continue
    else:
        # 对每个检测到的表盘进行处理
        for i in meter_list:
            cv2.imshow("det", i)
            cv2.waitKey(0)

            # 对表盘图像进行预处理
            image, _ = transform(i)
            # HWC -> CHW
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0)
            image = to_device(image)

            # 使用模型进行预测
            output = detector.detect1(image)

            # 提取预测结果
            pointer_pred, dail_pred, text_pred, preds, std_points = output['pointer'], output['dail'], output['text'], \
                                                                    output['reco'], output['std']

            # 解码预测的文本
            pred, preds_size = preds
            if pred != None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
            else:
                pred_transcripts = None

            # 将图像恢复到原始格式
            img_show = image[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

            # 使用表盘读数器获取最终结果
            result = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points)
