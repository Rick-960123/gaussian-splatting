import numpy as np  
import random
from matplotlib import pyplot as plt

Consistency = 3  #阶数
d = 0.9	#紧支域半径
dotNumber = 36	#散点数量
dotDinstance = 0.05	#散点间距

test = 0	# 0 为曲线，1 为直线

x = []
v = []
Vp = []
p = np.zeros((dotNumber, Consistency))
M = []


def w(xi,xj,d):
    s = abs(xi-xj)/d
    if(s <= 0.5):
        return (2/3)-4*s+4*s**2+4*s**3
    elif(s<=1):
        return (4/3)-4*s+4*s**2-(4/3)*s**3
    else:
        return 0

def b_compute(i, x, v, d):
    b = []
    for c in range(Consistency):
        t = 0
        for j in range(dotNumber):
            # if abs(x[i] - x[j]) < d:
                t +=  p[j][c] * w(x[i], x[j], d) * v[j]
        b.append(float(t))
    return b
        # for j in range(0, Consistency):
        #     w()


def A_compute(i, x, d):
    A = []
    for ci in range(Consistency):
        vec = []
        for cj in range(Consistency):
            t = 0
            for j in range(dotNumber):
                t +=  w(x[i], x[j], d) * p[j][cj] * p[j][ci]
            vec.append(float(t))
        A.append(vec)
    return A



def polySet(x):
    for i in range(dotNumber):
        for j in range(0, Consistency):
            p[i][j] = x[i]**(j)


def Vcontruct(a,x,i):
    value = 0
    for c in range(len(a)):
        value +=  a[c] * x[i]** c
    return value

def main():
    x = np.arange(-0.5*dotNumber*dotDinstance, 0.5*dotNumber*dotDinstance, dotDinstance)

    for i in range(0, dotNumber):
        if test == 0:
            v.append(i + random.randint(-50,50)/10)
            # v.append(random.randint(-50, 50) / 10)
        else:
            v.append(i + 0 / 10)

    polySet(x)

    for i in range(dotNumber):
        b = b_compute(i, x, v, d)

        A = A_compute(i, x, d)
        a = np.linalg.solve(A, b)
        Vp.append(Vcontruct(a, x, i))
        # V.append(a[0]+a[1]*x[i])

    plt.title("MLS")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x,v,color = "green", s= 10)
    plt.plot(x, Vp)
    plt.show()

if __name__ =='__main__':
    main()








