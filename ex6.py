from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X, y, h, w, targets = lfw.data, lfw.target, *lfw.images.shape[1:], lfw.target_names
print(f"Samples:{X.shape[0]}, Features:{X.shape[1]}, Classes:{len(targets)}")

# Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# PCA + transform
t0=time(); pca=PCA(n_components=150, svd_solver='randomized', whiten=True).fit(X_train)
X_train_pca,X_test_pca=pca.transform(X_train),pca.transform(X_test)
eigenfaces=pca.components_.reshape((150,h,w))
print(f"PCA done in {time()-t0:.3f}s")

# SVM
t0=time(); clf=SVC(kernel='rbf',class_weight='balanced',C=1000,gamma=0.001).fit(X_train_pca,y_train)
print(f"SVM trained in {time()-t0:.3f}s")

# Predict
t0=time(); y_pred=clf.predict(X_test_pca)
print(f"Prediction done in {time()-t0:.3f}s")
print(classification_report(y_test,y_pred,target_names=targets))
print(f"Accuracy: {accuracy_score(y_test,y_pred):.2f}\nConfusion:\n{confusion_matrix(y_test,y_pred)}")

# Plot
def plot_gallery(imgs,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1); plt.imshow(imgs[i].reshape(h,w),cmap='gray')
        plt.title(titles[i],size=12); plt.xticks(()); plt.yticks(())

plot_gallery(X_test,[f"pred:{targets[y_pred[i]].split()[-1]}\ntrue:{targets[y_test[i]].split()[-1]}" 
                    for i in range(y_pred.shape[0])],h,w)
plot_gallery(eigenfaces,[f"eigenface {i}" for i in range(eigenfaces.shape[0])],h,w)
plt.show()


#fisher faces
from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X, y, h, w, targets = lfw.data, lfw.target, *lfw.images.shape[1:], lfw.target_names
print(f"Samples:{X.shape[0]}, Features:{X.shape[1]}, Classes:{len(targets)}")

# Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# LDA (Fisherfaces)
n_comp=min(len(targets)-1,X_train.shape[1])
t0=time(); lda=LDA(n_components=n_comp).fit(X_train,y_train)
X_train_lda,X_test_lda=lda.transform(X_train),lda.transform(X_test)
print(f"LDA done in {time()-t0:.3f}s")

# SVM
t0=time(); clf=SVC(kernel='rbf',class_weight='balanced',C=1000,gamma=0.001).fit(X_train_lda,y_train)
print(f"SVM trained in {time()-t0:.3f}s")

# Predict
t0=time(); y_pred=clf.predict(X_test_lda)
print(f"Prediction done in {time()-t0:.3f}s")
print(classification_report(y_test,y_pred,target_names=targets))
print(f"Accuracy: {accuracy_score(y_test,y_pred):.2f}\nConfusion:\n{confusion_matrix(y_test,y_pred)}")

# Plot gallery
def plot_gallery(imgs,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(imgs[i].reshape(h,w),cmap='gray')
        plt.title(titles[i],size=12); plt.xticks(()); plt.yticks(())

plot_gallery(X_test,[f"pred:{targets[y_pred[i]].split()[-1]}\ntrue:{targets[y_test[i]].split()[-1]}" 
                    for i in range(y_pred.shape[0])],h,w)
plt.show()
