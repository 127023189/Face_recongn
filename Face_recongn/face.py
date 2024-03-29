# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog,QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSignal
from Face_identity import *
from model import *
import sys

class Ui_face_identity(QDialog):

    closemeg = pyqtSignal(object)

    def __init__(self,model=None,parent = None):
        super(Ui_face_identity,self).__init__(parent)

        self.model = model
        #初始化摄像头
        self.cap = cv2.VideoCapture()
        self.timer_camera = QtCore.QTimer()


        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.setui()
        self.connect()

        self.butsta = 0 # 默认是退出

        print(model)
        self.Face_identity = Face_identity(model = model)

    def setui(self):

        # 窗口
        self.setWindowTitle('face')

        # 显示视频

        self.label_show_camera = QtWidgets.QLabel(parent = self)
        self.label_show_camera.setFixedSize(640,480) # 通常是这个
        self.label_show_camera.move(20,10)
        self.label_show_camera.setAutoFillBackground(False)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_show_camera.setFont(font)
        self.label_show_camera.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.label_show_camera.setText("<html><head/><body><p align=\"center\">摄像区</p></body></html>")

        self.label_face = QtWidgets.QLabel(parent=self)
        self.label_face.move(700,10)
        self.label_face.setFixedSize(640,480)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_face.setFont(font)
        self.label_face.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.label_face.setText("<html><head/><body><p align=\"center\">图片识别区</p></body></html>")

        #设置按钮，主要是两个
        self.pushButton = QtWidgets.QPushButton(parent=self,text = '打开摄像头')
        #self.pushButton.move(210,500)
        self.pushButton.setGeometry(QtCore.QRect(210, 500, 131, 41))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.pushButton_2 = QtWidgets.QPushButton(parent=self,text = '退出')
        #self.pushButton_2.move(300,500)
        self.pushButton_2.setGeometry(QtCore.QRect(400, 500, 131, 41))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.pushButton_3 = QtWidgets.QPushButton(parent=self,text = '选择图片')
        self.pushButton_3.setGeometry(QtCore.QRect(750, 500, 131, 41))
        self.pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.label1 = QtWidgets.QLabel(parent=self,text = '结果')
        self.label1.setGeometry(QtCore.QRect(930, 500, 131, 41))



        self.label2 = QtWidgets.QLabel(parent=self)
        self.label2.setGeometry(QtCore.QRect(1000, 500, 131, 41))
        self.label2.setStyleSheet("background-color: rgb(255, 255, 255);")

    def connect(self): # 绑定事件
        self.pushButton.clicked.connect(lambda :[self.button_open_camera_click()])
        self.timer_camera.timeout.connect(lambda :self.show_camera())# 定时器任务
        self.pushButton_2.clicked.connect(lambda :[self.close_or_camera()])
        self.pushButton_3.clicked.connect(lambda :[self.openimage()])

    def openimage(self):
        try:
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.pgm;;*.jpg;;*.png;;All Files(*)")  # 在电脑上加代码
            jpg = QtGui.QPixmap(imgName).scaled(self.label_face.width()-100, self.label_face.height()-100)
            self.label_face.setPixmap(jpg)
            print("12")

            print(imgName)
            img = cv2.imread(imgName, 0)  # 灰度读入
            show_img = cv2.resize(img, (92, 112))  # ORL图像的大小
            pre_id = self.model.Prediction(show_img)
            print(pre_id)
            print("12")
            self.label2.setText(str(pre_id))

        except Exception as e:
            print(f"{e}")

    def button_open_camera_click(self):

        if self.timer_camera.isActive() == False: #开始启动
            ret = self.cap.open(self.CAM_NUM)#cv2.CAP_DSHOW
            print(ret)
            if ret == False:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(100)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.pushButton_2.setText('关闭摄像头')
                self.butsta = 1

            print("12")
        else: # 重复启动
            print(12)
            self.close_camera()
            self.pushButton.setText('打开摄像头')

    def show_camera(self):
        flag,self.image = self.cap.read()
        print("asasa")

        self.image = self.Face_identity.face_identity(self.image)
        print("wqwq")
        show_img = cv2.resize(self.image,(640,480))
        show_img = cv2.cvtColor(show_img,cv2.COLOR_BGR2RGB) # 格式转化,主要是转化为RGB格式
        showImage = QtGui.QImage(show_img.data, show_img.shape[1], show_img.shape[0],
                                 QtGui.QImage.Format_RGB888) # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage


    def close_camera(self):
        # 关闭摄像头
        if self.cap.isOpened():
            # 将一些视频流清空
            self.cap.release()
            self.label_show_camera.clear()
            print("关闭摄像头")
            self.pushButton_2.setText('退出')
            self.butsta = 0 # 关闭指令

            # 重新填充
            self.label_show_camera.setText("<html><head/><body><p align=\"center\">摄像区</p></body></html>")


        # 关闭定时器
        if self.timer_camera.isActive():
            self.timer_camera.stop()

    def close_or_camera(self):
        # 判断是关闭摄像头还是界面
        print(self.butsta)
        if self.butsta == 0:
            self.close()
        else:
            self.close_camera()


    def closeEvent(self,SCloseEvent):
        print(f"进入事件 ---- {SCloseEvent}")
        # 警告
        choice = QMessageBox.warning(None,"关闭","是否关闭",QMessageBox.Yes|QMessageBox.Cancel) # 确认和退出

        if choice == QMessageBox.Yes:

            self.closemeg.emit(None)

            # 关闭摄像头

            self.close_or_camera()

        else :
            # ignore
            SCloseEvent.ignore()

   # def keyPressEvent(self,QKeyEvent):
     #   if QKeyEvent.key() == QtCore.Qt.Key_Q:
        #       self.close_camera()


if __name__ == '__main__':

    App = QtWidgets.QApplication(sys.argv)
    model = train_SVM('att_faces')

    win = Ui_face_identity(model)
    win.show()
    sys.exit(App.exec_())
