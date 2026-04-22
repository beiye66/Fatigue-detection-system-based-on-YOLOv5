# 基于 YOLOv11 的疲劳驾驶检测系统

> 使用 YOLOv11 + PyQt5 实现的实时疲劳驾驶检测系统，支持摄像头、视频、图片三种检测模式，并内置连续帧过滤机制有效区分眨眼与真实闭眼。

---

## ✨ 功能特性

- **实时摄像头检测** — 实时识别驾驶员闭眼 / 打哈欠 / 正常状态
- **视频文件检测** — 支持 mp4、avi 等主流格式
- **图片检测** — 支持单张及批量图片（jpg/png/jpeg）
- **抗眨眼过滤** — 连续 15 帧（≈0.45s）才判定为疲劳，消除普通眨眼误报
- **疲劳告警** — 持续闭眼 / 打哈欠时画面顶部弹出红色告警条
- **结果记录** — 检测历史记录表格，支持截图保存
- **参数调节** — GUI 内可实时调整置信度和 NMS IOU 阈值

---

## 🎯 检测类别

| 类别 | 英文标签 | 说明 |
|------|---------|------|
| 闭眼 | `Eyeclosed` | 驾驶员闭眼状态 |
| 正常 | `Neutral` | 驾驶员正常睁眼 |
| 打哈欠 | `Yawn` | 驾驶员打哈欠 |

---

## 🛠️ 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.9 |
| PyTorch | 1.8.0 + CUDA 11.1（无 GPU 可用 CPU） |
| ultralytics | ≥ 8.3.0 |
| opencv-python | ≥ 4.5 |
| PyQt5 | 5.15.6 |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/beiye66/Fatigue-detection-system-based-on-YOLOv11.git
cd Fatigue-detection-system-based-on-YOLOv11
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv drowsiness_env
# Windows
drowsiness_env\Scripts\activate
# Linux / macOS
source drowsiness_env/bin/activate
```

### 3. 安装依赖

```bash
cd DrowsinessDetection
pip install -r requirements.txt
```

> ⚠️ 如使用 PyTorch 1.8.0，需将 NumPy 降级以避免兼容性问题：
> ```bash
> pip install "numpy<2"
> ```

### 4. 获取模型权重

模型权重文件（`weights/drowsiness-best.pt`）因体积较大未包含在仓库中，有两种方式获取：

**方式 A：自行训练（推荐）**

准备好数据集放入 `Drowsiness/images/` 和 `Drowsiness/labels/` 目录后运行：

```bash
python train.py
```

训练完成后将最佳权重复制到指定位置：

```bash
# Windows
copy runs\detect\drowsiness\weights\best.pt weights\drowsiness-best.pt
# Linux / macOS
cp runs/detect/drowsiness/weights/best.pt weights/drowsiness-best.pt
```

**方式 B：联系作者获取**

发邮件至 wb258770106@163.com 申请预训练权重。

### 5. 启动程序

```bash
python runMain.py
```

---

## 🖥️ 使用说明

启动后界面如下，各按钮功能说明：

| 按钮 | 功能 |
|------|------|
| 📂 打开文件 | 选择单张图片进行检测 |
| 📁 打开文件夹 | 批量检测文件夹内所有图片 |
| ▶ 视频检测 | 选择视频文件进行检测 |
| 💻 摄像头 | 开启实时摄像头检测 |
| 💾 保存 | 保存当前检测画面截图 |
| ⚙ 设置 | 调整置信度阈值和 NMS IOU 阈值 |
| 📄 模型 | 切换自定义模型权重 |

**疲劳告警说明：**

- 检测框实时显示每帧的分类结果（`Eyeclosed / Neutral / Yawn`）
- 当眼睛**连续闭合 ≥ 15 帧（约 0.45 秒）**时，画面左上角弹出红色告警条 `DROWSY: Eyes Closed`
- 当**连续打哈欠 ≥ 20 帧（约 0.6 秒）**时，弹出 `DROWSY: Yawning`
- 普通眨眼不触发告警

---

## 📂 项目结构

```
DrowsinessDetection/
├── Drowsiness/
│   ├── drowsiness.yaml      # 数据集配置
│   ├── label_name.py        # 类别中文名映射
│   ├── images/              # 训练集图片（本地，不含于仓库）
│   └── labels/              # 训练集标签（本地，不含于仓库）
├── utils/
│   └── __init__.py          # GUI 基类（窗口、绘图、表格等工具方法）
├── weights/
│   └── drowsiness-best.pt   # 模型权重（本地，不含于仓库）
├── DrowsinessDetecting.py   # 核心检测逻辑 + 主窗口
├── DrowsinessDetection_UI.py# Qt Designer 生成的 UI 布局
├── DrowsinessDetection_UI.ui# Qt Designer 源文件
├── collect_data.py          # 数据采集脚本（E/N/Y 键分类拍照）
├── prepare_dataset.py       # 数据集整理脚本（采集数据 → 训练集格式）
├── requirements.txt         # 依赖列表
├── runMain.py               # 程序入口
├── train.py                 # 模型训练脚本
└── yolo11s.pt               # YOLOv11s 预训练权重（本地，不含于仓库）
```

---

## 🔧 自采数据训练

如果想用自己的数据训练：

**Step 1：采集数据**

```bash
python collect_data.py
```

- `E` 键 → 保存闭眼样本
- `N` 键 → 保存正常样本
- `Y` 键 → 保存打哈欠样本
- `Q` 键 → 退出

**Step 2：整理数据集**

```bash
python prepare_dataset.py
```

自动将采集的图片整理为 YOLO 格式的 train/val/test 目录结构。

**Step 3：开始训练**

```bash
python train.py
```

默认配置：50 轮 + 早停（patience=10）、imgsz=640、batch=4，适合 4GB 显存 GPU。

---

## 📊 模型性能

训练环境：GTX 1650 Ti 4GB，YOLOv11s，数据集约 3000 张（3 类）

| 指标 | 数值 |
|------|------|
| mAP50 | ~93% |
| mAP50-95 | ~75.3% |
| 训练轮数 | 50（早停） |
| 推理速度 | ~30ms/帧（GPU） |

---

## 👤 作者

**Dayday_up**

📧 wb258770106@163.com
