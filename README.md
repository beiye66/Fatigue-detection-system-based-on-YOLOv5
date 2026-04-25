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

训练环境：GTX 1650 Ti 4GB，YOLOv11s，数据集约 3000 张（共3 类）

| 指标 | 数值 |
|------|------|
| mAP50 | ~93% |
| mAP50-95 | ~75.3% |
| 训练轮数 | 50（早停） |
| 推理速度（GPU） | ~30ms/帧 |
| 推理速度（Pi 4 NCNN） | ~8–15 FPS |

---

## 🍓 树莓派 4 部署

本项目已完整部署至树莓派 4，实现端侧实时疲劳检测，以下为完整流程。

### 阶段一：模型轻量化（PC 端）

树莓派无 GPU，直接运行 PyTorch `.pt` 模型帧率极低。使用 **NCNN** 格式（专为 ARM 架构优化）将模型压缩并加速：

```bash
# 在 Windows/PC 端执行
python -c "
from ultralytics import YOLO
m = YOLO('./weights/drowsiness-best.pt')
m.export(format='ncnn', imgsz=320, simplify=True)
"
```

> 导出过程中 ultralytics 会自动下载 PNNX 工具，通过 TorchScript → PNNX → NCNN 的转换链生成：
> - `model.ncnn.param`（网络结构）
> - `model.ncnn.bin`（权重）

### 阶段二：跨设备文件传输

通过局域网 SCP 将模型和项目文件推送到树莓派（在 PC 端新开一个终端，不是 SSH 窗口）：

```bash
# 传输整个项目目录
scp -r "DrowsinessDetection" pi@192.168.x.x:/home/pi/Fatigue/

# 单独传输 NCNN 模型文件夹
scp -r "DrowsinessDetection/weights/drowsiness-best_ncnn_model" pi@192.168.x.x:/home/pi/Fatigue/weights/
```

### 阶段三：树莓派端环境搭建

```bash
# 创建虚拟环境（隔离系统环境）
python3 -m venv ~/fatigue_env
source ~/fatigue_env/bin/activate

# 使用清华镜像源加速安装
pip install torch torchvision ultralytics ncnn opencv-python \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**空间不足的解决方法**（SD 卡分区未释放时常见）：

```bash
# 清理 pip 缓存
pip cache purge

# 指定临时目录到 SD 卡空间（绕过 /tmp 内存盘限制）
export TMPDIR=/home/pi/tmp
mkdir -p $TMPDIR
pip install ...
```

### 阶段四：端侧运行

树莓派通过 SSH 连接时没有 GUI，使用 `DISPLAY` 重定向将画面推送到外接显示屏：

```bash
# 将 OpenCV 画面输出到外接显示屏
export DISPLAY=:0
python detect_pi.py
```

**关键调优参数：**

| 参数 | 设定值 | 原因 |
|------|--------|------|
| 摄像头分辨率 | 640×480 | 降低采集开销，适配 Pi 4 算力 |
| 推理尺寸 | imgsz=320 | 比 640 快 4 倍，精度损失可接受 |
| NCNN 线程数 | 4 | 充分利用 Pi 4 的 4 核 CPU |

### 阶段五：时序状态机报警

相比 PC 端的帧数计数（15 帧），Pi 端改用**时间戳状态机**，不受帧率波动影响：

```python
import time

CLOSED_THRESHOLD = 2.0   # 闭眼超过 2 秒触发报警
YAWN_THRESHOLD   = 3.0   # 打哈欠超过 3 秒触发报警

closed_start = None
yawn_start   = None

# 每帧检测逻辑
if label == 'Eyeclosed':
    if closed_start is None:
        closed_start = time.time()
    elif time.time() - closed_start > CLOSED_THRESHOLD:
        print("⚠️  疲劳报警：持续闭眼！")
        # 在画面上叠加红色文字
else:
    closed_start = None
```

> 时间戳方案的优势：帧率从 8 FPS 变化到 15 FPS 时，阈值判断仍然准确；帧数方案在帧率不稳定时会产生偏差。

### 部署效果对比

| 运行环境 | 推理后端 | 帧率 |
|----------|---------|------|
| PC（GTX 1650 Ti） | PyTorch CUDA | ~33 FPS |
| 树莓派 4（PyTorch CPU） | PyTorch | ~1 FPS |
| 树莓派 4（NCNN） | NCNN ARM 优化 | ~8–15 FPS |

---

## 👤 作者

**Dayday_up**

📧 wb258770106@163.com
