import numpy as np

#array creation
arr=np.array([1,2,3,4,5,6,7,8,9,10])
print("array:",arr)
print("shape: ",arr.shape)
print("data type: ",arr.dtype)

#indexing and slicing 
arr1=np.array([10,20,30,40,50,60,70])

print("first element:",arr1[0])
print("last element:",arr1[-1])
print("slice: ",arr1[2:5])
print("every second element: ",arr1[::2])



#broadcasting
arr2=np.array([1,2,3,4,5])
result = arr2+10

print("original array:",arr2)
print("after adding 10:",result)
result1=result*2
print("after multiplying by 2:",result1)


#reshape
arr3=np.arange(1,13)
print("original array:",arr3)
matrix=arr3.reshape(3,4)
print("reshaped array:",matrix)
print("MATRIX SHAPE:",matrix.shape)

# mean and standard deviation 
arr3=np.array([[10,20,30],[40,50,60]])
print(arr3)
print("mean axis 0:",np.mean(arr3,axis=0))
print("mean axis 1:",np.mean(arr3,axis=1))

#vectorized vs loop

numbers=np.array([1,2,3,4,5])
squarevector=numbers**2
print("squarevector:",squarevector)

#looop
square_loop=[]
for number in numbers:
    square_loop.append(number**2)

print("using loop:",square_loop)


#multily
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])
result=np.matmul(A,B)
print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nMatrix Multiplication:")
print(result)


#boolean

numbers = np.array([10, 15, 20, 25, 30, 35, 40])
mask = numbers > 25

print("Boolean mask:")
print(mask)

print("\nValues greater than 25:")
print(numbers[mask])



#random module
np.random.seed(42)
random_numbers=np.random.randint(1,101,size=10)
print("Random integers:")
print(random_numbers)

#min max
data = np.array([45, 12, 78, 34, 90, 23, 56])

print("Original array:")
print(data)

print("\nMinimum:", np.min(data))
print("Maximum:", np.max(data))
print("Sum:", np.sum(data))
print("Average:", np.mean(data))

print("\nSorted array:")
print(np.sort(data))


