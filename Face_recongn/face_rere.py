# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face_rere.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_face_recon(object):
    def setupUi(self, face_recon):
        face_recon.setObjectName("face_recon")
        face_recon.resize(956, 600)
        face_recon.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(face_recon)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 640, 480))

        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(730, 380, 131, 41))
        self.pushButton.setStyleSheet("background-color: rgb(85, 85, 255);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(730, 440, 131, 41))
        self.pushButton_2.setStyleSheet("background-color: rgb(85, 85, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        face_recon.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(face_recon)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 956, 23))
        self.menubar.setObjectName("menubar")
        face_recon.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(face_recon)
        self.statusbar.setObjectName("statusbar")
        face_recon.setStatusBar(self.statusbar)

        self.retranslateUi(face_recon)
        QtCore.QMetaObject.connectSlotsByName(face_recon)

    def retranslateUi(self, face_recon):
        _translate = QtCore.QCoreApplication.translate
        face_recon.setWindowTitle(_translate("face_recon", "face_ewew"))
        self.label.setText(_translate("face_recon", "<html><head/><body><p align=\"center\">摄像头</p></body></html>"))
        self.pushButton.setText(_translate("face_recon", "打开摄像头"))
        self.pushButton_2.setText(_translate("face_recon", "关闭摄像头"))