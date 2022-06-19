import numpy as np

#Задание1
a = np.array([[1, 6], [2, 8], [3, 11], [3, 10], [1, 7]])

#Задание2
stroke = a.mean(axis=1)
row = a.mean(axis=0)
print (stroke, row)

stroke1 = stroke.mean()
row1 = row.mean()
a_mean = []
a_mean.append(stroke1)
a_mean.append(row1)
print (a_mean)

#Задание3

a_centered = np.subtract(a, mean_a)
print(np.subtract(a, mean_a))

#Задание4

a1 = a_centered[0:, 0:1].copy()
A1_t = np.transpose(a1)
A1_t = np.array(A1_t)
print(A1_t)

a2 = a_centered[0:, 1:].copy()
A2_t = np.transpose(a2)
A2_t = np.array(A2_t)
print(A2_t)

#"a_centered_sp = A1_t * A2_t
#print(a_centered_sp) В этом коде у меня получается векторное, а не скалярное произведение. Возможно это связано с тем, что неправильно взяты срезы, но как сделать по-другому программно, а не руками я не знаю(

x = [-4.2, -3.2, -2.2, -2.2, -4.2]
y = [0.8, 2.8, 5.8, 4.8, 1.8]

a_centered_sp = np.dot(x,y)
print(f'скалярное произведение столбцов: {a_centered_sp}\nскалярное произведение столбцов, разделенное на N-1 наблюдений: {a_centered_sp/4}')