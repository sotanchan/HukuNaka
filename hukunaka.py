import numpy as np
import random
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score
import glob
from sklearn.model_selection import train_test_split
import pickle
import cv2
import sys

test_images=glob.glob('test_images/*.*')
if len(test_images)==0:
    print('判別する画像がありません')
    sys.exit()
#予測モデルの読み込み
filename='hukushi_nakagawa.sav'
model = pickle.load(open(filename, 'rb'))
pca_reload = pickle.load(open("pca.pkl",'rb'))

nofaces=[]

amount=0
miss=[]
for test_image_name in test_images:
    try:

        im = Image.open(test_image_name)
        #openCVに変換
        ocv_im = np.asarray(im)
        #グレースケール変換
        image_gray = cv2.cvtColor(ocv_im, cv2.COLOR_BGR2GRAY)

        #カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

        color = (255, 255, 255) #白



        # 検出した場合
        answer=''
        if len(facerect) ==1:
            amount+=1
            #検出した顔を囲む矩形の作成
            for rect in facerect:
                    cv2.rectangle(ocv_im, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),color, thickness=2)
                    top=tuple(rect)[1]-5
                    bottom=top+tuple(rect)[3]+15
                    left=tuple(rect)[0]-5
                    right=left+tuple(rect)[2]+15
                    face=ocv_im[top:bottom,left:right]

                    #検出した顔をPILに変換
                    test_face = Image.fromarray(face)
                    sample_resize =test_face.resize((128,128))
                    #画像を配列に変換
                    sample_array = np.ravel(np.asarray(sample_resize))
                    sample_regularized = sample_array/255

                    sample_regularized=sample_regularized.reshape(1,len(sample_regularized))
                    sample_regularized=pca_reload.transform(sample_regularized)


                    #予測
                    test_label = model.predict(sample_regularized)

                    if(test_label==1):
                        answer='福士蒼汰'


                    else:
                        answer='中川大志'
                    
            print(f"ファイル名：{test_image_name}\n判定結果：{answer}\n")
        else:
            nofaces.append(test_image_name)
            
    except:
        miss.append(test_image_name)
        

print()

if len(nofaces)>0:
    print("顔が判定されなかった画像")
    for i in nofaces:
        print(i)

print()

if len(miss)>0:
    print("処理が正常に行われなかった画像")
    for i in miss:
        print(i)
