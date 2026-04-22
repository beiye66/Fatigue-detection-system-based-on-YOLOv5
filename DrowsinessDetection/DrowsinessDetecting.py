# -*- coding: utf-8 -*-
"""
运行本项目需要 Python 3.9+ 及以下依赖库（完整库见 requirements.txt）：
    ultralytics>=8.3.0
    opencv-python==4.5.5.64
    PyQt5==5.15.6
点击运行主程序 runMain.py，程序所在文件夹路径中请勿出现中文
"""
import os
import random
import time
from os import getcwd

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from ultralytics import YOLO

from Drowsiness.label_name import Chinese_name
from utils import QMainWindow


class Drowsiness_MainWindow(QMainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(Drowsiness_MainWindow, self).__init__(*args, **kwargs)
        self.author_flag = False

        self.setupUi(self)
        self.retranslateUi(self)
        self._apply_default_icons()
        self.setUiStyle(window_flag=True, transBack_flag=True)

        self.path = getcwd()
        self.video_path = getcwd()

        self.timer_camera = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()
        self.flag_timer = ""

        self.LoadModel()
        self.slot_init()
        self.files = []
        self.cap_video = None
        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture(self.CAM_NUM)

        self.detInfo = []
        self.current_image = []
        self.detected_image = None
        self.count = 0
        self.res_set = []
        self.c_video = 0
        self.count_name = ["闭眼", "正常", "打哈欠"]
        self.count_table = []

        # 连续帧过滤：只有眼睛持续闭合一段时间才算疲劳（过滤眨眼）
        # 正常眨眼约 100–400ms；定时器间隔 30ms，因此 15 帧 ≈ 0.45s
        self.EYECLOSED_SUSTAIN_FRAMES = 15
        self.YAWN_SUSTAIN_FRAMES = 20
        self.eyeclosed_streak = 0
        self.yawn_streak = 0

    def slot_init(self):
        self.toolButton_file.clicked.connect(self.choose_file)
        self.toolButton_folder.clicked.connect(self.choose_folder)
        self.toolButton_video.clicked.connect(self.button_open_video_click)
        self.timer_video.timeout.connect(self.show_video)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.toolButton_model.clicked.connect(self.choose_model)
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)
        self.tableWidget.cellPressed.connect(self.table_review)
        self.toolButton_saveing.clicked.connect(self.save_file)
        self.toolButton_settings.clicked.connect(self.setting)
        self.toolButton_author.clicked.connect(self.disp_website)
        self.toolButton_version.clicked.connect(self.disp_version)

    def table_review(self, row, col):
        try:
            if col == 0:
                this_path = self.tableWidget.item(row, 1)
                res = self.tableWidget.item(row, 2)
                axes = self.tableWidget.item(row, 3)

                if (this_path is not None) & (res is not None) & (axes is not None):
                    this_path = this_path.text()
                    if os.path.exists(this_path):
                        res = res.text()
                        axes = axes.text()

                        image = self.cv_imread(this_path)
                        image = cv2.resize(image, (850, 500))

                        axes = [int(i) for i in axes.split(",")]
                        confi = float(self.tableWidget.item(row, 4).text())

                        count = self.count_table[row]
                        self.label_numer_result.setText(str(sum(count)))
                        image = self.drawRectEdge(image, axes, alpha=0.2, addText=res+" "+str(round(confi * 100, 2))+"%")
                        self.display_image(image)

                        self.label_xmin_result.setText(str(int(axes[0])))
                        self.label_ymin_result.setText(str(int(axes[1])))
                        self.label_xmax_result.setText(str(int(axes[2])))
                        self.label_ymax_result.setText(str(int(axes[3])))
                        self.label_score_result.setText(str(round(confi * 100, 2)) + "%")
                        self.label_class_result.setText(res)

                        QtWidgets.QApplication.processEvents()
        except:
            self.label_display.setText('重现表格记录时出错，请检查表格内容！')
            self.label_display.setStyleSheet("background-color: rgb(220, 220, 220); border: 1px solid rgb(180, 180, 180);")

    def LoadModel(self, model_path=None):
        """加载 YOLOv11 预训练模型"""
        weight = model_path if model_path else './weights/drowsiness-best.pt'
        self.conf_thres = 0.25  # 置信度阈值
        self.iou_thres = 0.45   # NMS IoU 阈值
        self.model = YOLO(weight)
        self.names = self.model.names  # {0: 'Eyeclosed', 1: 'Neutral', 2: 'Yawn'}
        color = [[132, 56, 255], [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
                 [72, 249, 10], [146, 204, 23], [61, 219, 134], [26, 147, 52], [0, 212, 187],
                 [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49],
                 [44, 153, 168], [0, 194, 255], [52, 69, 147], [100, 115, 255], [0, 24, 236]]
        self.colors = color if len(self.names) <= len(color) else [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def predict(self, image):
        """对单帧 BGR 图像进行推理，返回 (result, 耗时秒)"""
        t1 = time.time()
        results = self.model(image, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        t2 = time.time()
        return results[0], round(t2 - t1, 2)

    def _apply_default_icons(self):
        """把 .ui 里引用的自定义图标（:/images/icons/...）统一替换成 Qt 自带
        标准图标 + 纯文字标签。即使 pyuic5 重新生成 UI 文件也不会丢效果。"""
        style = self.style()
        SP = QtWidgets.QStyle

        # 主窗口：清掉窗口图标 + 背景图 stylesheet
        self.setWindowIcon(QtGui.QIcon())
        self.setStyleSheet("")

        btn_icons = {
            'toolButton_settings': SP.SP_FileDialogDetailedView,
            'toolButton_file':     SP.SP_DialogOpenButton,
            'toolButton_camera':   SP.SP_ComputerIcon,
            'toolButton_version':  SP.SP_MessageBoxInformation,
            'toolButton_saveing':  SP.SP_DialogSaveButton,
            'toolButton_video':    SP.SP_MediaPlay,
            'toolButton_author':   SP.SP_MessageBoxQuestion,
            'toolButton_model':    SP.SP_FileIcon,
            'toolButton_folder':   SP.SP_DirIcon,
        }
        for attr, sp in btn_icons.items():
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setIcon(style.standardIcon(sp))

        label_texts = {
            'label_picNumber':   "目标:",
            'label_picConf':     "置信:",
            'label_picTime':     "耗时:",
            'label_picSelect':   "选择:",
            'label_picResult':   "结果:",
            'label_picLocation': "位置:",
        }
        for attr, text in label_texts.items():
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.setStyleSheet("")
                lbl.setText(text)
                lbl.setAlignment(QtCore.Qt.AlignCenter)

        if hasattr(self, 'label_display'):
            self.label_display.setStyleSheet(
                "background-color: rgb(220, 220, 220); border: 1px solid rgb(180, 180, 180);"
            )
            self.label_display.setText("请选择图片/视频/摄像头开始检测")
            self.label_display.setAlignment(QtCore.Qt.AlignCenter)

    def _reset_fatigue_state(self):
        self.eyeclosed_streak = 0
        self.yawn_streak = 0

    def _update_fatigue_state(self, frame_class_names):
        """根据本帧出现的类别累计连续帧。返回当前触发的告警列表 [(name, streak), ...]。

        规则：
          - 检测到 Eyeclosed → 闭眼连续帧 +1；检测到 Neutral → 重置为 0
          - 检测到 Yawn → 打哈欠连续帧 +1；检测到 Neutral → 重置为 0
          - 未检测到任何类别（漏检一帧）→ 保持不变，避免把真实疲劳冲掉
        """
        has_eyeclosed = 'Eyeclosed' in frame_class_names
        has_yawn = 'Yawn' in frame_class_names
        has_neutral = 'Neutral' in frame_class_names

        if has_eyeclosed:
            self.eyeclosed_streak += 1
        elif has_neutral:
            self.eyeclosed_streak = 0

        if has_yawn:
            self.yawn_streak += 1
        elif has_neutral:
            self.yawn_streak = 0

        alerts = []
        if self.eyeclosed_streak >= self.EYECLOSED_SUSTAIN_FRAMES:
            alerts.append(('Eyes Closed', self.eyeclosed_streak))
        if self.yawn_streak >= self.YAWN_SUSTAIN_FRAMES:
            alerts.append(('Yawning', self.yawn_streak))
        return alerts

    def _draw_fatigue_banner(self, image, alerts):
        """在画面顶部绘制疲劳告警条（红底白字）。alerts 为空时原样返回。"""
        if not alerts:
            return image
        for i, (name, streak) in enumerate(alerts):
            y = 15 + i * 42
            text = f"DROWSY: {name} ({streak}f)"
            cv2.rectangle(image, (10, y), (360, y + 34), (0, 0, 200), -1)
            cv2.putText(image, text, (18, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def choose_model(self):
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()

        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有目标')
        self.clearUI()
        self.flag_timer = ""

        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "选取图片文件", getcwd(),
                                                                "Model File (*.pt)")
        if fileName_choose != '':
            self.toolButton_model.setToolTip(fileName_choose + ' 已选中')
        else:
            fileName_choose = None
            self.toolButton_model.setToolTip('使用默认模型')
        self.LoadModel(fileName_choose)

    def select_obj(self):
        QtWidgets.QApplication.processEvents()
        if self.flag_timer == "video":
            self.timer_video.start(30)
        elif self.flag_timer == "camera":
            self.timer_camera.start(30)

        ind = self.comboBox_select.currentIndex() - 1
        ind_select = ind
        if ind <= -1:
            ind_select = 0
        if len(self.detInfo) > 0:
            self.label_class_result.setText(self.detInfo[ind_select][0])
            self.label_score_result.setText(str(self.detInfo[ind_select][2]))
            self.label_xmin_result.setText(str(int(self.detInfo[ind_select][1][0])))
            self.label_ymin_result.setText(str(int(self.detInfo[ind_select][1][1])))
            self.label_xmax_result.setText(str(int(self.detInfo[ind_select][1][2])))
            self.label_ymax_result.setText(str(int(self.detInfo[ind_select][1][3])))

        image = self.current_image.copy()
        if len(self.detInfo) > 0:
            for i, box in enumerate(self.detInfo):
                if ind != -1:
                    if ind != i:
                        continue
                label = '%s %.0f%%' % (box[0], float(box[2]) * 100)
                self.label_score_result.setText(box[2])
                image = self.drawRectBox(image, box[1], addText=label, color=self.colors[box[3]])
            self.display_image(image)

    def choose_folder(self):
        self.timer_camera.stop()
        self.timer_video.stop()
        self.c_video = 0
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()

        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有目标')
        self.clearUI()
        self.flag_timer = ""

        dir_choose = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", self.path)
        self.path = dir_choose
        if dir_choose != "":
            self.textEdit_pic.setText(dir_choose + '文件夹已选中')
            self.label_display.setText('正在启动识别系统...\n\nleading')
            QtWidgets.QApplication.processEvents()

            rootdir = os.path.join(self.path)
            for (dirpath, dirnames, filenames) in os.walk(rootdir):
                for filename in filenames:
                    temp_type = os.path.splitext(filename)[1]
                    if temp_type in ('.png', '.jpg', '.jpeg'):
                        img_path = dirpath + '/' + filename
                        image = self.cv_imread(img_path)
                        image = cv2.resize(image, (850, 500))
                        self.current_image = image.copy()

                        result, InferenceNms = self.predict(image)
                        self.label_time_result.setText(str(InferenceNms))

                        boxes = result.boxes
                        im0 = image.copy()
                        count = [0 for _ in self.count_name]
                        self.detInfo = []

                        if boxes is not None and len(boxes):
                            for box in reversed(boxes):
                                xyxy = [int(x) for x in box.xyxy[0].tolist()]
                                c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                name = self.names[cls]

                                self.detInfo.append([name, [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, cls])
                                self.label_class_result.setText(name)
                                self.label_score_result.setText('%.2f' % conf)
                                self.label_xmin_result.setText(str(c1[0]))
                                self.label_ymin_result.setText(str(c1[1]))
                                self.label_xmax_result.setText(str(c2[0]))
                                self.label_ymax_result.setText(str(c2[1]))

                                for cn in range(len(self.count_name)):
                                    if name == self.count_name[cn]:
                                        count[cn] += 1

                                label = '%s %.0f%%' % (name, conf * 100)
                                im0 = self.drawRectBox(im0, xyxy, alpha=0.2, addText=label, color=self.colors[cls])

                                res_all = [name, conf, [c1[0], c1[1], c2[0], c2[1]]]
                                self.res_set.append(res_all)
                                self.change_table(img_path, res_all[0], res_all[2], res_all[1])

                            for _ in range(len(boxes)):
                                self.count_table.append(count)
                            self.label_numer_result.setText(str(sum(count)))

                            self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                            self.comboBox_select.clear()
                            self.comboBox_select.addItem('所有目标')
                            for i in range(len(self.detInfo)):
                                self.comboBox_select.addItem("{}-{}".format(self.detInfo[i][0], i + 1))
                            self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                            image = im0.copy()
                        else:
                            self.label_numer_result.setText("0")
                            self.label_class_result.setText('0')
                            self.label_score_result.setText("0")
                            self.label_xmin_result.setText("0")
                            self.label_ymin_result.setText("0")
                            self.label_xmax_result.setText("0")
                            self.label_ymax_result.setText("0")

                        self.detected_image = image.copy()
                        self.display_image(image)
                        QtWidgets.QApplication.processEvents()
        else:
            self.clearUI()

    def choose_file(self):
        """图像检测"""
        self.timer_camera.stop()
        self.timer_video.stop()
        self.c_video = 0
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()

        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有目标')
        self.clearUI()
        self.flag_timer = ""

        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,
            "图片(*.jpg;*.jpeg;*.png)")
        self.path = fileName_choose

        if fileName_choose != '':
            self.flag_timer = "image"
            self.textEdit_pic.setText(fileName_choose + '文件已选中')
            self.label_display.setText('正在启动识别系统...\n\nleading')
            QtWidgets.QApplication.processEvents()

            image = self.cv_imread(self.path)
            image = cv2.resize(image, (850, 500))
            self.current_image = image.copy()

            result, InferenceNms = self.predict(image)
            self.label_time_result.setText(str(InferenceNms))

            boxes = result.boxes
            im0 = image.copy()
            count = [0 for _ in self.count_name]
            self.detInfo = []

            if boxes is not None and len(boxes):
                for box in reversed(boxes):
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = self.names[cls]

                    self.detInfo.append([name, [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, cls])
                    self.label_class_result.setText(name)
                    self.label_score_result.setText('%.2f' % conf)
                    self.label_xmin_result.setText(str(c1[0]))
                    self.label_ymin_result.setText(str(c1[1]))
                    self.label_xmax_result.setText(str(c2[0]))
                    self.label_ymax_result.setText(str(c2[1]))

                    for cn in range(len(self.count_name)):
                        if name == self.count_name[cn]:
                            count[cn] += 1

                    label = '%s %.0f%%' % (name, conf * 100)
                    im0 = self.drawRectBox(im0, xyxy, alpha=0.2, addText=label, color=self.colors[cls])

                    res_all = [name, conf, [c1[0], c1[1], c2[0], c2[1]]]
                    self.res_set.append(res_all)
                    self.change_table(self.path, res_all[0], res_all[2], res_all[1])

                for _ in range(len(boxes)):
                    self.count_table.append(count)
                self.label_numer_result.setText(str(sum(count)))

                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.detInfo)):
                    self.comboBox_select.addItem("{}-{}".format(self.detInfo[i][0], i + 1))
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                image = im0.copy()
            else:
                self.label_numer_result.setText("0")
                self.label_class_result.setText('0')
                self.label_score_result.setText("0")
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

            self.detected_image = image.copy()
            self.display_image(image)
        else:
            self.clearUI()

    def button_open_video_click(self):
        self.c_video = 0
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if self.cap:
            self.cap.release()

        self.clearUI()
        QtWidgets.QApplication.processEvents()

        if not self.timer_video.isActive():
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    self.video_path,
                                                                    "视频(*.mp4;*.avi)")
            self.video_path = fileName_choose

            if fileName_choose != '':
                self.flag_timer = "video"
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.setStyleText(self.textEdit_video)

                self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                try:
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                except:
                    print("[INFO] could not determine # of frames in video")

                self._reset_fatigue_state()
                self.timer_video.start(30)
            else:
                self.flag_timer = ""
                self.clearUI()
        else:
            self.flag_timer = ""
            self.timer_video.stop()
            self.cap_video.release()
            self.label_display.clear()
            time.sleep(0.5)
            self.clearUI()
            self.comboBox_select.clear()
            self.comboBox_select.addItem('所有目标')
            QtWidgets.QApplication.processEvents()

    def show_video(self):
        flag, image = self.cap_video.read()
        if flag:
            image = cv2.resize(image, (850, 500))
            self.current_image = image.copy()

            result, useTime = self.predict(image)
            self.label_time_result.setText(str(useTime))
            QtWidgets.QApplication.processEvents()

            boxes = result.boxes
            im0 = image.copy()
            count = [0 for _ in self.count_name]
            self.detInfo = []
            frame_class_names = []

            if boxes is not None and len(boxes):
                for box in reversed(boxes):
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = self.names[cls]
                    frame_class_names.append(name)

                    self.detInfo.append([name, [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, cls])
                    self.label_class_result.setText(name)
                    self.label_score_result.setText('%.2f' % conf)
                    self.label_xmin_result.setText(str(c1[0]))
                    self.label_ymin_result.setText(str(c1[1]))
                    self.label_xmax_result.setText(str(c2[0]))
                    self.label_ymax_result.setText(str(c2[1]))

                    for cn in range(len(self.count_name)):
                        if name == self.count_name[cn]:
                            count[cn] += 1

                    label = '%s %.0f%%' % (name, conf * 100)
                    im0 = self.drawRectBox(im0, xyxy, addText=label, color=self.colors[cls])

                    self.c_video += 1
                    if self.c_video % 10 == 0:
                        res_all = [name, conf, [c1[0], c1[1], c2[0], c2[1]]]
                        self.res_set.append(res_all)
                        self.change_table(str(self.count), res_all[0], res_all[2], res_all[1])
                        self.count_table.append(count)

                self.label_numer_result.setText(str(sum(count)))

                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.detInfo)):
                    self.comboBox_select.addItem("{}-{}".format(self.detInfo[i][0], i + 1))
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                image = im0.copy()
            else:
                self.label_numer_result.setText("0")
                self.label_class_result.setText('0')
                self.label_score_result.setText("0")
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

            alerts = self._update_fatigue_state(frame_class_names)
            image = self._draw_fatigue_banner(image, alerts)

            self.detected_image = image.copy()
            QtWidgets.QApplication.processEvents()
            self.display_image(image)
        else:
            self.timer_video.stop()

    def button_open_camera_click(self):
        self.c_video = 0
        if self.timer_video.isActive():
            self.timer_video.stop()
        QtWidgets.QApplication.processEvents()

        if self.cap_video:
            self.cap_video.release()

        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                              u"请检测相机与电脑是否连接正确！ ",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                self.flag_timer = "camera"
                self.clearUI()
                self.textEdit_camera.setText('实时摄像已启动')
                self.setStyleText(self.textEdit_camera)
                self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()
                self._reset_fatigue_state()
                self.timer_camera.start(30)
        else:
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.clearUI()
            QtWidgets.QApplication.processEvents()

    def show_camera(self):
        flag, image = self.cap.read()
        if flag:
            self.current_image = image.copy()

            result, useTime = self.predict(image)
            self.label_time_result.setText(str(useTime))

            boxes = result.boxes
            im0 = image.copy()
            count = [0 for _ in self.count_name]
            self.detInfo = []
            frame_class_names = []

            if boxes is not None and len(boxes):
                for box in reversed(boxes):
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = self.names[cls]
                    frame_class_names.append(name)

                    self.detInfo.append([name, [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, cls])
                    self.label_class_result.setText(name)
                    self.label_score_result.setText('%.2f' % conf)
                    self.label_xmin_result.setText(str(c1[0]))
                    self.label_ymin_result.setText(str(c1[1]))
                    self.label_xmax_result.setText(str(c2[0]))
                    self.label_ymax_result.setText(str(c2[1]))

                    for cn in range(len(self.count_name)):
                        if name == self.count_name[cn]:
                            count[cn] += 1

                    label = '%s %.0f%%' % (name, conf * 100)
                    im0 = self.drawRectBox(im0, xyxy, addText=label, color=self.colors[cls])

                    self.c_video += 1
                    if self.c_video % 10 == 0:
                        res_all = [name, conf, [c1[0], c1[1], c2[0], c2[1]]]
                        self.res_set.append(res_all)
                        self.change_table(str(self.count), res_all[0], res_all[2], res_all[1])
                        self.count_table.append(count)

                self.label_numer_result.setText(str(sum(count)))

                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.detInfo)):
                    self.comboBox_select.addItem("{}-{}".format(self.detInfo[i][0], i + 1))
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                image = im0.copy()
            else:
                self.label_numer_result.setText("0")
                self.label_class_result.setText('0')
                self.label_score_result.setText("0")
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

            alerts = self._update_fatigue_state(frame_class_names)
            image = self._draw_fatigue_banner(image, alerts)

            self.detected_image = image.copy()
            self.display_image(image)
        else:
            self.timer_video.stop()

    def save_file(self):
        if self.detected_image is not None:
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            cv2.imwrite('./pic_' + str(now_time) + '.png', self.detected_image)
            QMessageBox.about(self.centralwidget, "保存文件", "\nSuccessed!\n文件已保存！")
        else:
            QMessageBox.about(self.centralwidget, "保存文件", "saving...\nFailed!\n请先选择检测操作！")
