import numpy as np
from PIL import Image, ImageOps
import glob
from sklearn.model_selection import train_test_split
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier



#顔画像の枚数
num_of_hukushi= int(len(glob.glob('福士蒼汰顔/福士蒼汰顔*.jpg')))
#顔ない画像の枚数
num_of_nakagawa = int(len(glob.glob('中川大志顔/中川大志顔*.jpg')))
#学習データの全枚数
num_of_all_data = num_of_hukushi + num_of_nakagawa

N_col = 128*128*3 # 行列の列数
X_train = np.zeros((num_of_all_data, N_col)) # 学習データ格納のためゼロ行列生成
y_train = np.zeros((num_of_all_data)) # 学習データに対するラベルを格納するためのゼロ行列生成
i_count=0
hukushi_faces=glob.glob('福士蒼汰顔/福士蒼汰顔*.jpg')
nakagawa_faces = glob.glob("中川大志顔/中川大志顔*.jpg") 


def img_to_arr(image_list,X_train,y_train,i_count,label):
    for im in image_list:
        im = Image.open(im).convert('RGB')
 
        img_resize =im.resize((128,128))
        #画像を配列に変換
        im_array = np.ravel(np.asarray(img_resize))
        im_regularized = im_array/255.

        X_train[i_count,:] =  im_regularized
        y_train[i_count] = label
        i_count += 1
    
    return X_train,y_train,i_count

X_train,y_train,i_count=img_to_arr(hukushi_faces,X_train,y_train,i_count,0)
X_train,y_train,i_count=img_to_arr(nakagawa_faces,X_train,y_train,i_count,1)


pca=PCA(100)
X_train = pca.fit_transform(X_train)
pickle.dump(pca, open("pca.pkl","wb"))


X_train, X_test ,y_train,y_test= train_test_split(X_train,y_train, test_size=0.1,random_state=0)


x=range(1,len(y_train))
y=[]
max_score=0
max_K=0
for i in x:
    model = KNeighborsClassifier(n_neighbors = i)  
    model.fit(X_train, y_train)# モデルの学習
    y.append(model.score(X_test, y_test))
    if max_score<=model.score(X_test, y_test):
        max_score=model.score(X_test, y_test)
        max_K=i


print('最高正答率：{}'.format(str(max_score)))
print('最高正答率のときのK：{}'.format(str(max_K)))


model = KNeighborsClassifier(n_neighbors = max_K)  
model.fit(X_train, y_train)# モデルの学習



filename = 'hukushi_nakagawa.sav'
pickle.dump(model, open(filename, 'wb'))

