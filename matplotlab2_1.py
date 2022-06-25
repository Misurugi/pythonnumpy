import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import math

#Задание 1
x = np.arange(1, 8, 1)
print (x)
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]
print (y)
plt.plot(x, y)
plt.show()
plt.scatter(x, y)
plt.show(

#Задание 2
t = np.linspace(1, 10, 51)
print(t)
f = np.cos(t)
print(f)
plt.plot(t, f, color = 'green')

title_font = {
    "fontsize": 15,
    "fontweight": "bold",
    "color": "Green",
    "family": "serif"
   }
label_font = {
    "fontsize": 12,
    "family": "serif",
}
plt.title("График f(t)", fontdict=title_font)
plt.xlabel("Значения t", fontdict=label_font)
plt.ylabel("Значения f", fontdict=label_font)
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show()

#Задание3
x = np.linspace(-3, 3, 51)
print(t)
y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)
fig = plt.figure(figsize=(8, 6))
ax_1 = fig.add_subplot(2, 2, 1)
plt.plot(x, y1, '-')
ax_2 = fig.add_subplot(2, 2, 2)
plt.plot(x, y2, '*')
ax_3 = fig.add_subplot(2, 2, 3)
plt.plot(x, y3, '--')
ax_4 = fig.add_subplot(2, 2, 4)
plt.plot(x, y4, '.')

ax_1.set(xlim = [-5, 5],
       title = 'y1')
ax_2.set(title = 'y2')
ax_3.set(title = 'y3')
ax_4.set(title = 'y4')

#Задания на повторение
#1
a=np.arange(12, 24)
print(a)

#2
a1=a.reshape(2, 6)
a2=a.reshape(3, 4)
a3=a.reshape(6, 2)
a4=a.reshape(4, 3)
a5=a.reshape(12, 1)
print(a1,a2,a3,a4,a5)

#3
a1=a.reshape(-1, 6)
a2=a.reshape(3, -1)
a3=a.reshape(6, -1)
a4=a.reshape(-1, 3)
a5=a.reshape(12, -1)
print(a1, a2, a3, a4, a5)

#5
b=np.random.randn(3,4)
print(b)

print(b.size)
b=b.flatten()
print(b.size)
print(b)

#6
a=np.arange(20, 0, -2)
print(a)

#7
b=np.arange(20, 1, -2).reshape(1,10)
print(b)

a = np.zeros((2, 2))
print(a)

b = np.zeros((3, 2))+1
print(b)

#8
v = np.vstack((a, b))
print(v)
a.shape
b.shape
v.shape
v.size

#9
a=np.arange(0, 12)
print(a)

A=a.reshape(4,3)
print(a)


At=A.T
print(At)

B = np.dot(A, At)
print(B)

B.shape

B_inv=np.linalg.inv(B)
B_inv

#10
np.random.seed(42)

#11
c=np.random.randint(0, 16,16)
print(c)

#12
C=c.reshape(4,4)
print(C)

D=B+C*10
print(D)

d=np.linalg.det(D)
print(d)

r=np.linalg.matrix_rank(D)
print(r)

D_inv=np.linalg.inv(D)
print(D_inv)

#13
D_inv=np.where(D_inv<0, 0,1)
print(D_inv)

E=np.where(D_inv==1, B, C)
print(E)




