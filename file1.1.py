import numpy as np
arr=np.array([[10,20,30],[40,50,60],[70,80,90]])

print("overall mean:",np.mean(arr))

print("mean by columns:",np.mean(arr,axis=0))

print("mean by row:",np.mean(arr,axis=1))

print("overall standard daviation",np.std(arr))

print("standard daviation by column: ",np.std(arr,axis0))
print("sstandard daviation by row: ",np.std(arr,axis=1))

