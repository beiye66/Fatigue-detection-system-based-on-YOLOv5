# 基于YOLOv5的疲劳驾驶检测系统

## 📖 项目简介

本项目是一个基于YOLOv5深度学习算法的疲劳驾驶检测系统，采用Python和PyQt5开发，具备完整的图形用户界面。系统能够实时检测驾驶员的疲劳状态，包括闭眼、打哈欠等行为，有效预防因疲劳驾驶引发的交通事故。

## ✨ 主要功能

- 🔍 **实时疲劳检测** - 支持摄像头实时检测驾驶员疲劳状态
- 📷 **图片检测** - 支持单张或多张图片批量检测
- 🎥 **视频检测** - 支持视频文件中的疲劳行为检测
- 📊 **结果统计** - 检测结果可视化展示和统计分析
- 🔐 **用户认证** - 支持用户登录和注册功能
- 💾 **数据保存** - 检测结果可保存为文件

## 🎯 检测类别

系统能够检测以下三种疲劳状态：
- **闭眼 (Eyeclosed)** - 驾驶员闭眼状态
- **正常 (Neutral)** - 驾驶员正常状态
- **打哈欠 (Yawn)** - 驾驶员打哈欠状态

## 🛠️ 技术栈

### 深度学习框架
- **YOLOv5** - 目标检测算法
- **PyTorch 1.8.0** - 深度学习框架
- **TensorFlow 2.9.1** - 机器学习框架
- **Keras 2.9.0** - 深度学习API

### 图像处理
- **OpenCV 4.5.5.64** - 计算机视觉库
- **scikit-image 0.19.3** - 图像处理库
- **Pillow 9.0.1** - 图像处理库

### 图形界面
- **PyQt5 5.15.6** - 图形用户界面框架

### 其他依赖
- **scipy 1.8.0** - 科学计算库
- **numpy** - 数值计算库

## 📁 项目结构

```
Fatigue detection/
└── DrowsinessDetection/
    ├── Drowsiness/                    # 数据集和配置文件
    │   ├── images/                   # 图像数据集
    │   │   ├── train/                # 训练集图像
    │   │   ├── test/                 # 测试集图像
    │   │   └── valid/                # 验证集图像
    │   ├── drowsiness.yaml           # YOLO配置文件
    │   └── label_name.py             # 标签名称定义
    ├── .idea/                        # PyCharm IDE配置
    ├── runMain.py                    # 程序主入口
    ├── DrowsinessDetecting.py        # 主要检测逻辑
    ├── DrowsinessDetection_UI.py    # 主界面UI代码
    ├── DrowsinessLoginUI.py          # 登录界面
    ├── DrowsinessDetection_UI.ui    # Qt Designer界面文件
    ├── train.py                      # 模型训练脚本
    ├── test.py                       # 通用测试脚本
    ├── testVideo.py                  # 视频测试脚本
    ├── testPicture.py                # 图片测试脚本
    └── RecSystem_rc.py              # 资源文件
```

## 🚀 快速开始

### 环境要求

- **Python版本**: 3.8
- **操作系统**: Windows/Linux/macOS
- **硬件要求**: 支持CUDA的GPU（推荐）或CPU

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <项目地址>
   cd "Fatigue detection"
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

   如果没有requirements.txt文件，请手动安装以下依赖：
   ```bash
   pip install opencv-python==4.5.5.64
   pip install tensorflow==2.9.1
   pip install PyQt5==5.15.6
   pip install scikit-image==0.19.3
   pip install torch==1.8.0
   pip install keras==2.9.0
   pip install Pillow==9.0.1
   pip install scipy==1.8.0
   ```

3. **运行程序**
   ```bash
   cd DrowsinessDetection
   python runMain.py
   ```

   **重要提示**: 程序所在文件夹路径中请勿出现中文字符

### 使用说明

1. **登录系统**
   - 默认账号: `admin`
   - 默认密码: `123456`
   - 测试账号: `test`
   - 测试密码: `123456`

2. **功能选择**
   - **摄像头检测**: 点击摄像头按钮开始实时检测
   - **图片检测**: 选择图片文件进行检测
   - **视频检测**: 选择视频文件进行检测
   - **批量检测**: 选择文件夹进行批量检测

3. **结果查看**
   - 检测结果会在界面中实时显示
   - 可以查看检测统计信息
   - 支持结果保存和导出

## 🔧 模型训练

如果您需要训练自定义模型：

1. **准备数据集**
   - 按照YOLO格式组织数据集
   - 将图像放入对应的train/test/valid文件夹
   - 确保标注文件格式正确

2. **配置参数**
   - 修改`Drowsiness/drowsiness.yaml`文件
   - 设置训练参数和路径

3. **开始训练**
   ```bash
   python train.py
   ```

## 📊 数据集信息

项目包含完整的数据集，专门用于疲劳驾驶检测任务：
- **训练集**: 包含大量标注好的面部图像
- **测试集**: 用于模型性能评估
- **验证集**: 用于训练过程中的验证

## 🎨 界面展示

系统提供友好的图形用户界面：
- 现代化的UI设计
- 实时视频流显示
- 检测结果可视化
- 统计图表展示
- 中文本地化界面

## 🔍 技术特点

1. **高效检测** - 基于YOLOv5算法，检测速度快
2. **高准确率** - 经过大量数据训练，检测准确率高
3. **实时性能** - 支持实时视频流处理
4. **用户友好** - 图形化界面，操作简单
5. **可扩展性** - 支持自定义模型训练

## 📝 注意事项

- 确保Python环境为3.8版本
- 程序路径中不要包含中文字符
- 首次运行可能需要下载预训练模型
- 建议使用GPU以获得更好的性能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目遵循MIT许可证。

## 🙏 致谢

感谢以下开源项目的支持：
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

**作者**: 思绪无限  
**博客**: [https://www.cnblogs.com/sixuwuxian/](https://www.cnblogs.com/sixuwuxian/)  
**知乎**: [https://www.zhihu.com/people/sixuwuxian](https://www.zhihu.com/people/sixuwuxian)  
**Bilibili**: 思绪亦无限  
**公众号**: AI技术研究与分享