"""
desc:
    因为logistic回归中在最关键的调参步骤中需要用到梯度上升法去逼近目标参数，
    梯度上升（梯度下降也是类似）从字面理解是比较容易的，百度搜一下相关的解释，
    大部分人都会理解。难在从数学层面去表示这个过程，理解这个才能用程序去实现。
    下面举一个一元二次函数的例子去解释梯度上升法求极大值。
"""
# f(x) = -x**2 + 3*x -1 可视化
import matplotlib.pyplot as plt 
import numpy as np 
# X = np.arange(-8,11,0.1)
# Y = -X**2 + 3*X - 1
# plt.plot(X,Y)
# plt.show()


import math
W0 = np.arange(-8,11,0.1)
W1 = np.arange(-8,11,0.1)
Y = W1*2.5+W0
Z = np.exp(-Y)
sigmoid = np.log10(1/(1+Z))
plt.plot(W0,sigmoid)
plt.show()
"""
    对于上述函数，如果要求函数的极大值，只需要对f(x)求导,让f'(x)=0,其x就是所得解。
    但如果应用梯度上升法求这个函数的极大值,则是需要用一个增量一点点去逼近这个极大值。
    f'(x)=-2*x+3,假设当前点为x=-5,f(-5)=-41。如果通过一个增量获取到的f(-5+△x)>f(-5),
    则将当前点移动到x=-5+△x.通过迭代直到f(x+△x)<f(x'+△x),此时的x即为梯度上升法逼近的极大值
"""
def f(x):# 原函数
    return -x**2 + 3*x - 1
def αf(x):# f(x)的导数
    return -2*x+3 
def gradAscent0(x):
    β = 0.01 # αY只是斜率,如果要获取增量的话需要乘以△x,这里用β表示△x
    current_x = x # 设置当前点
    while 1:
        current_y = f(current_x)
        next_x = current_x + β # 试探点
        next_y = f(current_x) + β*αf(current_x) #关键语句
        if next_y > current_y:
            current_x = next_x
        else:
            break
    print("f(x)=-x**2 +3*x-1 的极值点为：",current_x)

"""
    上面的函数只适用于当前点在对称轴左侧，如果当前点在对称轴右侧，
    由于右侧是单调递减的，函数会认为当前点就是极值点，下面为改进的函数
"""
def gradAscent1(x):
    β = 0.01 # αY只是斜率,如果要获取增量的话需要乘以△x,这里用β表示△x
    current_x = x # 设置当前点
    if αf(current_x)>0:
        while 1:
            current_y = f(current_x)
            next_x = current_x + β # 试探点
            next_y = f(current_x) + β*αf(current_x) #关键语句
            if next_y > current_y:
                current_x = next_x
            else:
                break
    else:
        while 1:
            current_y = f(current_x)
            next_x = current_x - β # 试探点
            next_y = f(current_x) - β*αf(current_x) #关键语句
            if next_y > current_y:
                current_x = next_x
            else:
                break
    print("f(x)=-x**2 +3*x-1 的极值点为：",current_x)

# if __name__ == '__main__':
    # gradAscent0(5)
    # gradAscent1(5)


"""
注释:
    上述例子中的关键点在于评估下一个取值的策略，即next_y = f(current_x) - β*αf(current_x)，
    在梯度上升法中，这个策略可以描述为：f(x+1) = f(x)+β*[αf(x)/x],其中αf(x)/x为梯度，在连续
    可导的一元函数中可以解释为函数的导数。
    在连续可导的一元函数中，函数的某一个点的导数就是切线斜率；但是在连续可导的多元函数中，
    函数的某一个点可能有多条切线（想下球面），而梯度上升法的目标就是从众多的切线中取最抖的一条
    切线，以保证在一个单位的β下β*[αf(x)/x]变化最大。
"""