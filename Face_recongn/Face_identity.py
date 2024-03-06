import cv2

class Face_identity(object):

    def __init__(self,model = None):

        self.model = model


    #先得到人脸框的位置,这里要最大阈值化
    def face_detection(self,img):

        #导入相应的数据

        face_cascade = cv2.CascadeClassifier(r"C:\Users\yixi\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml") # 需要修改，将其换成cv2对应文件的地址

        # 进行人脸检测

        faces = face_cascade.detectMultiScale(img, 1.2, 5)

        #这里先把最大矩形框位置找出来

        # 初始
        max_area = 0
        h1,w1,x1,y1 = 0,0,0,0
        print(f"人脸可能坐标 ： {faces}")
        for (x,y,w,h) in faces:
            # 画矩阵框
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if w*h > max_area :
                max_area = w*h
                x1,y1,w1,h1 = x,y,w,h

        return img,x1,y1,w1,h1

    #识别程序
    def face_identity(self,img):

        # 导入相应的数据

        face_cascade = cv2.CascadeClassifier(
            r"C:\Users\yixi\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

        # 进行人脸检测
        faces = face_cascade.detectMultiScale(img, 1.2, 5)


        print(f"人脸可能坐标 ： {faces}")

        try:

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

                img_identity = img[y-5:y+h+5,x-5:x+w+10] # 可以适当扩大 x是行行坐标，y是列坐标,适当扩大一下

                # 灰度化
                img_identity = cv2.cvtColor(img_identity,cv2.COLOR_BGR2GRAY)

                #调整图像大小
                img_identity = cv2.resize(img_identity,(92,112)) # ORL图像的大小

                # 进行识别并将文字显示在图像框上
                if(self.model == None):
                    img_id = '无模型'
                    print("无模型")
                else:
                     try:
                         img_id = self.model.Prediction(img_identity)
                         cv2.putText(img, str(img_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)

                     except Exception as e:

                         print(f"{e}")
        except Exception as e:
            print(f"{e}")


        print("1212")

        return img


if __name__ == '__main__':
    Face_identity().show_img(None)