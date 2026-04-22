# -*- coding: utf-8 -*-
"""
YOLOv11 疲劳驾驶检测模型训练脚本

运行前确保已安装依赖：
    pip install ultralytics>=8.3.0

使用方法：
    python train.py

训练完成后权重保存在：
    runs/detect/drowsiness/weights/best.pt
    runs/detect/drowsiness/weights/last.pt

将 best.pt 复制到 weights/drowsiness-best.pt 后运行主程序。
"""

from ultralytics import YOLO


def train():
    # 加载 YOLOv11s 预训练模型（s=small，速度快，适合树莓派部署）
    # 可选: yolo11n.pt(最小) / yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt(最大)
    model = YOLO('yolo11s.pt')

    model.train(
        data='./Drowsiness/drowsiness.yaml',   # 数据集配置
        epochs=50,                              # 训练轮数
        imgsz=640,                              # 输入图像尺寸
        batch=4,                                # 批次大小（GTX 1650 Ti 4GB显存安全值）
        workers=2,                              # 数据加载线程数
        device=0,                               # 使用GPU训练，无GPU改为 'cpu'
        project='runs/detect',                  # 结果保存目录
        name='drowsiness',                      # 实验名称
        exist_ok=True,                          # 允许覆盖已有结果
        pretrained=True,                        # 使用预训练权重（迁移学习）
        optimizer='SGD',                        # 优化器
        lr0=0.01,                               # 初始学习率
        patience=10,                            # 早停：连续10轮mAP无提升则停止训练
        save=True,                              # 保存权重
        val=True,                               # 每轮验证
        plots=True,                             # 保存训练曲线图
    )

    print("\n训练完成！")
    print("最佳权重路径: runs/detect/drowsiness/weights/best.pt")
    print("请将 best.pt 复制到 weights/drowsiness-best.pt 以供主程序使用：")
    print("  copy runs\\detect\\drowsiness\\weights\\best.pt weights\\drowsiness-best.pt")


if __name__ == '__main__':
    train()
