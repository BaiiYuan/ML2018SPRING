import sys
import csv
import numpy as np
from skimage import io,transform
EigenNUM=4
Shape = 600

folder_name = "./"+sys.argv[1]+"/"
def Trans(M):
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype(np.uint8)
	M = M.reshape((Shape, Shape, 3))
	return M
X=[]
print("loading...")
for cou in range(415):
	file_name = folder_name+str(cou)+".jpg"
	# print(file_name)
	img = io.imread(file_name)/255.
	img = img.flatten()
	X.append(img)
X = np.asarray(X)
## Average
Mean_X = np.mean(X, axis=0)
# io.imsave("./image/mean.jpg", np.reshape(Mean_X*255, (Shape, Shape, 3)).astype(np.uint8))


print("SVD...")
U, s, V = np.linalg.svd((X*255 - Mean_X*255).T, full_matrices=False)
# np.save("svd_U.npy", U)
# np.save("svd_s.npy", s)
# np.save("svd_V.npy", V)

# U, s, V = np.load("svd_U.npy"), np.load("svd_s.npy"), np.load("svd_V.npy")
# print(U.shape, s.shape, V.shape)
# print(s.tolist())

# for i in range(4):
# 	print("No.%d-->%f" %(i, s[i]/np.sum(s)))


# ## eigenface
# for i in range(11):
# 	M = U[:,i:i+1].reshape((-1))
# 	M = Trans(M)
# 	file_name="./image/eigenface"+str(i+1)+".jpg"
# 	io.imsave(file_name, -M)


print(U[:,0:EigenNUM])
## Reconstruct
y_name = sys.argv[2]
file_name = folder_name+y_name
print(file_name)
img = io.imread(file_name)/255.
y = img.flatten()
y -= Mean_X
weight = np.dot(y.T, U[:,0:EigenNUM])
print(weight)
R = np.dot(U[:,0:EigenNUM], weight.T)
R += Mean_X
R = Trans(R)
print(R.shape)
io.imsave("./reconstruction.jpg", R)