import numpy as np
import itertools as itt
import matplotlib.pyplot as plt

n = 2

b = np.zeros(n)

phi_1 = lambda x: 1
phi_2 = lambda x: 1/x
phi_s = [phi_1, phi_2]

datContent = [i.strip().split() for i in open("farmakoloski.dat").readlines()]
x_s = []
y_s = []
dy_s = []

for line in datContent[1:]:
	x_s.append(float(line[0]))
	y_s.append(float(line[1]))
	dy_s.append(float(line[2]))

x_s = np.array(x_s)
y_s = np.array(y_s)
dy_s = np.array(dy_s)


y_tr_s = 1/y_s
dy_tr_s = dy_s/y_s**2


def make_A1():
	A = np.zeros((n, n))
	for j, k in itt.product(range(n), range(n)):
		for l in range(len(x_s)):
			A[j, k] += phi_s[j](x_s[l])*phi_s[k](x_s[l]) / dy_tr_s[l]**2
	return A

def make_b1():
	b = np.zeros(n)
	for j in range(n):
		for l in range(len(x_s)):
			b[j] += y_tr_s[l] * phi_s[j](x_s[l]) / dy_tr_s[l]**2
	return b

A = make_A1()
b = make_b1()
a = np.linalg.solve(A, b)
A_inv = np.linalg.inv(A)
da_0 = np.sqrt(A_inv[0][0])
da_1 = np.sqrt(A_inv[1][1])
print(a)
y_0 = 1/a[0]
dy_0 = da_0/a[0]**2
a_0 = a[1]/a[0]
da_0 = da_1/a[0] + da_0*a[1]/a[0]**2
print(y_0, dy_0, 'in', a_0, da_0)

t_s = np.linspace(x_s[0], x_s[-1], 1000)


#lt.plot(x_s, y_s, 'x')
plt.plot(t_s, y_0*t_s/(t_s+np.ones_like(t_s)*a_0))
plt.plot(t_s, (y_0+dy_0)*t_s/(t_s+np.ones_like(t_s)*(a_0-da_0)), 'k--', linewidth=1, alpha=0.3)
plt.plot(t_s, (y_0-dy_0)*t_s/(t_s+np.ones_like(t_s)*(a_0+da_0)), 'k--', linewidth=1, alpha=0.3)
plt.fill_between(t_s, (y_0+dy_0)*t_s/(t_s+np.ones_like(t_s)*(a_0-da_0)), (y_0-dy_0)*t_s/(t_s+np.ones_like(t_s)*(a_0+da_0)), color='orange', alpha=0.2)
plt.errorbar(x_s, y_s, yerr=dy_s, fmt='o', capsize=3)
plt.xscale('log')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

#plot errorja
'''
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

def f(x_s, y_0, a, p):
	return y_0*x_s**p / (x_s**p + a**p)

def fun_min(args):
	y_0 = args[0]
	a = args[1]
	p = args[2]
	return (y_s - f(x_s, y_0, a, p)) / dy_s

res = least_squares(fun_min, [100, 20, 1], method='lm')
res_curve, M_cov = curve_fit(f, x_s, y_s, p0=[100, 20, 1], sigma=dy_s, absolute_sigma=True, method='lm')


plt.plot(t_s, abs(f(t_s, *res.x)-y_0*t_s/(t_s+np.ones_like(t_s)*a_0)), alpha=0.6)
#plt.plot(t_s, f(t_s, *res.x), color='green')
#plt.plot(t_s, y_0*t_s/(t_s+np.ones_like(t_s)*a_0), color='red')
plt.xscale('log')
plt.ylabel('y')
plt.xlabel('x')
plt.show()'''