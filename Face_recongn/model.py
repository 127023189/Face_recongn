# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""


import numpy as np 
import skimage 
import cv2
from sklearn.decomposition import PCA
import os
from sklearn import svm 
import random
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

class train_SVM:
    
	#nCellX, nCellY = 8, 4  # 将图像划分为 nCellX*nCellY 个子区域

    def __init__(self,path,size = 0.7): # 初始化
        self.path = path
        self.size = size


    def training(self):
        # 先得到数据
        X_train, y_train, X_test, y_test = self.load_data()

        # 先分别提取测试集与训练集的特征
        print("Start extracting features ...")

        print("当前提取训练集 ：\n")
        feature_X_train = self.getfeatures(X_train)
        print("当前提取测试集 : \n")
        feature_X_test = self.getfeatures(X_test)

        print("\nStart PCA dimensionality reduction ...")
        # 接下来进行PCA降维
        pca_X_train, pca_X_test = self.Pca(feature_X_train, feature_X_test)
        self.pca_X_train,self.y_train = pca_X_train,y_train
        self.pca_X_test,self.y_test = pca_X_test,y_test
        print("\nStart training the model ...")
        self.model = self.modelSVM(pca_X_train, y_train)

        """
        print(f"\n训练集 : \n")
        self.test(pca_X_train,y_train)
        print(f"\n测试集 ： \n")
        self.test(pca_X_test,y_test)
        """

    def basicLBP(self,gray,R=2,P=8): # 使用p=8,r=2的算子
        num,height,width = gray.shape
        dst = np.zeros((num,height* width))
        for i in range(num):
            # 使用比较原始的LBP算子,并拉直
            dst[i,:] = skimage.feature.local_binary_pattern(gray[i,:], P, R,'default').flatten()
            if (i+1) % 20 == 0:
                print(f"提取到图片 {i+1}")
        # 转置
        return dst.T

    def calLBPHistogram(self,imgLBP,nCellX = 8,nCellY = 4): # 获取统计直方图
        # 92 = 4*23,112 = 8*14 
        img = imgLBP.reshape(112,92)
        height,width = np.shape(img)
        # 分为8*4份
        hCell,wCell = height//nCellX,width//nCellY
        LBPHistoGram = np.zeros((256,nCellX*nCellY)) # 第一个维度为cell数，第二个是LBP值
        
        for h in range(nCellX):
            for w in range(nCellY):
                # 使用block 
                cell = img[(h*hCell):(h+1)*hCell,(w*wCell):(w+1)*wCell]
                cell = np.array(cell,np.uint8)
                
                # 选择掩膜方法
               # mask = np.zeros(np.shape(img),np.uint8) 
                # 掩膜部分是需要选择的
                #mask[(h*hCell):(h+1)*hCell,(w*wCell):(w+1)*wCell] = 255
                
                # 统计直方图
                hist = cv2.calcHist([cell], [0], None, [256], [0, 255])
                LBPHistoGram[:,h*nCellY+w] =  hist.flatten().T # h从0开始,计算累计了多少block

        #print(LBPHistoGram.shape)
        return LBPHistoGram.flatten().T

    def modelSVM(self,Xtrain,ytrain): # 训练模型
        
        smodel = svm.SVC(kernel = 'rbf',probability = True) # 采取交叉验证,5折
        smodel.fit(Xtrain,ytrain)
        return smodel

    def load_data(self): # 读取数据
        """数据集来自ORL的40*10张灰度图片，每张图片大小为92×112"""
        X_train = []
        X_test = []
        y_test = []
        y_train = []

        prename = os.listdir(self.path)
        line = int(10*self.size)

        for dirname in prename: # 循环每个文件
            for i in range(1,line+1): # 8-2分配
                namepath = self.path + '/'+dirname + '/' +str(i)+ '.pgm' # 获取文件位置
                
                img = cv2.imread(namepath,0)# 灰度读入
                
                X_train.append(img)
                y_train.append(str(dirname))
            
            for i in range(line+1,11):
                namepath = self.path + '/'+dirname +'/' + str(i) + '.pgm' # 获取文件位置
                img = cv2.imread(namepath,0)# 灰度读入
                X_test.append(img)
                y_test.append(str(dirname))
        
        #这里再将训练集打乱
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(X_train)
        random.seed(randnum)
        random.shuffle(y_train)
        #print(f"测试集 标签")
        #print(y_test)
        print(f"训练集大小 ： {len(X_train)} 测试集大小 ： {len(X_test)}")
        #化为矩阵，标签转置
        return np.array(X_train),np.array(y_train).T,np.array(X_test),np.array(y_test).T

    def getfeatures(self,input): # 提取特征
        # 这里input是包含多张照片的
        inputLBP = self.basicLBP(input)

        # 获得直方图
        LBPHistogram = np.zeros((256*8*4,np.shape(inputLBP)[1])) # 这里列表示图片数目

        for i in range(np.shape(inputLBP)[1]):# 遍历每一张图片
            blockHistogram = self.calLBPHistogram(inputLBP[:,i])
            LBPHistogram[:,i] = blockHistogram

        # 转置
        return LBPHistogram.T

    def Pca(self,X_train,X_test,n_components = 200): # 保存200个特征
        #randomized:适用于数据量大,数据维度多同时主成分数目比例又较低的 PCA 降维
        self.pca = PCA(n_components = n_components,svd_solver = 'randomized',whiten = True) 
        # 测试集使用训练fit数据
        self.pca.fit(X_train)
        pcaX_train = self.pca.transform(X_train)
        pcaX_test = self.pca.transform(X_test)
        return pcaX_train,pcaX_test

    def test(self,X_test,y_test): #测试当前模型
        y_pre = self.model.predict(X_test)

        # acc正确率
        self.acc = accuracy_score(y_test,y_pre)
        print(f"acc :\n{self.acc}")

        # 精度
        self.pre_score = precision_score(y_test,y_pre,average='micro')
        print(f"precision_score :\n{self.pre_score}")

        # 召回率
        self.recall = recall_score(y_test,y_pre,average='micro')
        print(f"recall_score :\n{self.recall}")

    # 预测单张图片
    def Prediction(self,input):
        img = self.getfeatures(np.array([input])) # 注意是三维
        img = self.pca.transform(img)
        pre = self.model.predict(img)[0] # 只输出标签
        return pre



