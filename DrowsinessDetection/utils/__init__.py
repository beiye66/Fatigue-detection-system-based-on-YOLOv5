# -*- coding: utf-8 -*-
"""
utils/__init__.py
替代原加密版本的工具基类，提供 Drowsiness_MainWindow 所需的全部辅助方法。

包含以下功能：
  - 窗口初始化与样式设置
  - 支持中文路径的图像读取
  - OpenCV 图像显示到 QLabel
  - 半透明检测框绘制
  - 历史记录表格操作
  - UI 状态清除
  - 设置、版本、作者弹窗
"""

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow as MainWindow,
    QTableWidgetItem,
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
)

from DrowsinessDetection_UI import Ui_MainWindow


class QMainWindow(MainWindow, Ui_MainWindow):
    """
    自定义基类，通过多重继承把 PyQt5 原生窗口（MainWindow）
    和 Qt Designer 生成的界面布局（Ui_MainWindow）合并在一起，
    并在此基础上提供检测系统所需的全套辅助方法。
    """

    def __init__(self, *args, obj=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_pos = QPoint()    # 无边框窗口拖拽起点
        self._frameless = False      # 是否启用了无边框模式
        # 防止 QImage 数据被 GC 回收的持有引用
        self._last_qimage_buffer = None

    # =========================================================================
    # 一、窗口初始化
    # =========================================================================

    def showTime(self):
        """
        显示主窗口。
        runMain.py 调用 win.showTime() 而不是 win.show()，
        这里做等价实现，有需要可以在 show() 前加启动动画等逻辑。
        """
        self.show()

    def setUiStyle(self, window_flag=True, transBack_flag=True):
        """
        设置窗口样式。

        为了稳定性，默认实现保留系统标题栏（可见、可关闭、可拖动）。
        如果想要无边框+透明背景的异形外观，请把下面的 SAFE_MODE 改成 False。

        window_flag=True    → 尝试应用自定义窗口样式
        transBack_flag=True → 尝试开启背景透明
        """
        SAFE_MODE = True   # ← 改成 False 可启用完全无边框模式

        if SAFE_MODE:
            # 安全模式：保留系统标题栏，确保窗口一定可见、可关闭
            return

        if window_flag:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self._frameless = True
        if transBack_flag:
            self.setAttribute(Qt.WA_TranslucentBackground, True)

    # 无边框窗口拖拽支持（用类方法 override，PyQt5 推荐做法）-------------------
    def mousePressEvent(self, event):
        if self._frameless and event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._frameless and event.buttons() == Qt.LeftButton \
                and not self._drag_pos.isNull():
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._frameless:
            self._drag_pos = QPoint()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """无边框模式下按 Esc 关闭窗口"""
        if self._frameless and event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    # =========================================================================
    # 二、图像读取
    # =========================================================================

    def cv_imread(self, filePath):
        """
        支持中文路径的图像读取。
        cv2.imread() 在 Windows 中遇到中文路径会返回 None，
        改用 np.fromfile + imdecode 绕过这个问题。
        """
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        if cv_img is None:
            return None
        # 如果是 4 通道（带 Alpha），只取前三个通道
        if len(cv_img.shape) > 2 and cv_img.shape[2] > 3:
            cv_img = cv_img[:, :, :3]
        return cv_img

    # =========================================================================
    # 三、图像显示
    # =========================================================================

    def display_image(self, img):
        """
        将 OpenCV BGR 格式的图像显示到界面中央的 label_display。

        流程：
          BGR numpy array
            → 转 RGB（Qt 使用 RGB 顺序）
            → 构造 QImage
            → 构造 QPixmap 并等比缩放适应 label 大小
            → 设置给 label_display
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 保证数据在内存中连续，且由 self 持有引用防止 GC
        img_rgb = np.ascontiguousarray(img_rgb)
        self._last_qimage_buffer = img_rgb
        h, w, c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label_display.setPixmap(
            pixmap.scaled(
                self.label_display.width(),
                self.label_display.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # =========================================================================
    # 四、检测框绘制
    # =========================================================================

    def drawRectBox(self, img, xyxy, alpha=0.2, addText='', color=(132, 56, 255)):
        """
        绘制带半透明填充的检测框。

        参数：
          img     - 原始 BGR 图像（numpy array）
          xyxy    - 检测框坐标，支持列表/元组/tensor，格式 [x1,y1,x2,y2]
          alpha   - 填充透明度（0=完全透明，1=完全不透明），默认 0.2
          addText - 框上方显示的文字（类别 + 置信度）
          color   - BGR 颜色元组

        实现思路：
          1. 把填充色块画到 overlay（原图副本）上
          2. 用 addWeighted 把 overlay 以 alpha 权重叠加回原图 → 半透明效果
          3. 在叠加结果上再画实线边框（不透明）
          4. 在边框左上角画文字背景 + 白色文字
        """
        x1 = int(xyxy[0]); y1 = int(xyxy[1])
        x2 = int(xyxy[2]); y2 = int(xyxy[3])

        # 半透明填充
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 实线边框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

        # 文字标签
        if addText:
            font_scale = 0.55
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(
                addText, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
            )
            # 文字背景矩形（紧贴框的左上角，向上延伸）
            ty = max(y1 - 2, th + 4)
            cv2.rectangle(
                img,
                (x1, ty - th - 4),
                (x1 + tw + 2, ty),
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img, addText,
                (x1 + 1, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )
        return img

    def drawRectEdge(self, img, axes, alpha=0.2, addText=''):
        """
        绘制检测框（供表格回放时调用）。
        axes 格式为列表 [x1, y1, x2, y2]，颜色固定为紫色。
        复用 drawRectBox 实现，避免重复代码。
        """
        color = (132, 56, 255)
        return self.drawRectBox(img, axes, alpha=alpha, addText=addText, color=color)

    # =========================================================================
    # 五、历史记录表格
    # =========================================================================

    def change_table(self, path, res, axes, conf):
        """
        向历史记录表格（tableWidget）末尾追加一行检测结果。

        列顺序：序号 | 画面标识 | 结果 | 位置 | 置信度

        参数：
          path  - 图片路径（图片/文件夹模式）或帧编号（视频/摄像头模式）
          res   - 检测类别名（中文）
          axes  - 检测框坐标列表 [x1, y1, x2, y2]
          conf  - 置信度浮点数（0~1）
        """
        # 首次写入前，清除 Qt Designer 预置的空白占位行
        if self.tableWidget.rowCount() > 0:
            first_item = self.tableWidget.item(0, 2)
            if first_item is None or first_item.text() == '':
                self.tableWidget.setRowCount(0)

        row = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row)

        def _cell(text, align=Qt.AlignCenter):
            item = QTableWidgetItem(str(text))
            item.setTextAlignment(align)
            return item

        self.tableWidget.setItem(row, 0, _cell(self.count))
        self.tableWidget.setItem(row, 1, _cell(str(path), Qt.AlignLeft | Qt.AlignVCenter))
        self.tableWidget.setItem(row, 2, _cell(str(res)))
        coords = ','.join([str(int(a)) for a in axes])
        self.tableWidget.setItem(row, 3, _cell(coords))
        self.tableWidget.setItem(row, 4, _cell(f'{float(conf):.4f}'))

        self.count += 1
        self.tableWidget.scrollToBottom()

    # =========================================================================
    # 六、UI 状态清除
    # =========================================================================

    def clearUI(self):
        """
        把界面上所有动态显示区域重置为初始状态。
        在切换输入模式（图片→视频、视频→摄像头等）前调用，
        避免上一次的结果残留在界面上。
        """
        # 主显示区域：清除图片，恢复默认背景
        self.label_display.clear()
        self.label_display.setStyleSheet(
            "background-color: rgb(220, 220, 220); border: 1px solid rgb(180, 180, 180);"
        )
        self.label_display.setText("请选择图片/视频/摄像头开始检测")
        self.label_display.setAlignment(Qt.AlignCenter)

        # 右侧信息面板
        self.label_class_result.setText('0')
        self.label_score_result.setText('0')
        self.label_xmin_result.setText('0')
        self.label_ymin_result.setText('0')
        self.label_xmax_result.setText('0')
        self.label_ymax_result.setText('0')
        self.label_numer_result.setText('0')
        self.label_time_result.setText('0 s')

        # 清空表格并重置计数
        self.tableWidget.setRowCount(0)
        self.count = 0
        self.count_table = []

    # =========================================================================
    # 七、文本框样式
    # =========================================================================

    def setStyleText(self, textEdit):
        """
        将状态提示文本框的文字颜色改为高亮青色，
        用于"摄像头已启动"、"视频已选中"等提示信息。
        """
        textEdit.setStyleSheet("color: #0dceda;")

    # =========================================================================
    # 八、弹窗（作者/版本/设置）
    # =========================================================================

    def disp_website(self):
        """显示开发者信息弹窗（对应 toolButton_author 按钮）"""
        QMessageBox.about(
            self.centralwidget,
            "开发信息",
            "基于 YOLOv11 的疲劳驾驶检测系统\n\n"
            "作者: Dayday_up\n"
            "邮箱: wb258770106@163.com",
        )

    def disp_version(self):
        """显示版本信息弹窗（对应 toolButton_version 按钮）"""
        QMessageBox.about(
            self.centralwidget,
            "版本信息",
            "Drowsiness Detection  v2.0\n\n"
            "框架：YOLOv11 (Ultralytics) + PyQt5\n"
            "环境：Python 3.9 | PyTorch 1.8\n"
            "检测类别：闭眼 / 正常 / 打哈欠",
        )

    def setting(self):
        """
        打开参数设置对话框（对应 toolButton_settings 按钮）。
        提供置信度阈值和 NMS IOU 阈值的滑动条调节，
        点击确定后实时生效（下一帧推理即使用新参数）。
        """
        dlg = QDialog(self.centralwidget)
        dlg.setWindowTitle("参数设置")
        dlg.setFixedSize(340, 200)

        layout = QVBoxLayout()

        # --- 置信度阈值 ---
        lbl_conf = QLabel(f"置信度阈值 (conf_thres): {self.conf_thres:.2f}")
        sld_conf = QSlider(Qt.Horizontal)
        sld_conf.setRange(1, 95)
        sld_conf.setValue(int(self.conf_thres * 100))
        # 滑动时实时更新标签文字
        sld_conf.valueChanged.connect(
            lambda v: lbl_conf.setText(f"置信度阈值 (conf_thres): {v / 100:.2f}")
        )
        layout.addWidget(lbl_conf)
        layout.addWidget(sld_conf)

        # --- NMS IOU 阈值 ---
        lbl_nms = QLabel(f"NMS IOU 阈值 (iou_thres):  {self.iou_thres:.2f}")
        sld_nms = QSlider(Qt.Horizontal)
        sld_nms.setRange(1, 95)
        sld_nms.setValue(int(self.iou_thres * 100))
        sld_nms.valueChanged.connect(
            lambda v: lbl_nms.setText(f"NMS IOU 阈值 (iou_thres):  {v / 100:.2f}")
        )
        layout.addWidget(lbl_nms)
        layout.addWidget(sld_nms)

        # --- 按钮行 ---
        btn_row = QHBoxLayout()
        btn_ok     = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        def _apply():
            self.conf_thres = sld_conf.value() / 100
            self.iou_thres  = sld_nms.value()  / 100
            dlg.accept()

        btn_ok.clicked.connect(_apply)
        btn_cancel.clicked.connect(dlg.reject)

        dlg.setLayout(layout)
        dlg.exec_()
