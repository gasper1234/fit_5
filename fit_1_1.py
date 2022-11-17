import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit


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

def f(x_s, y_0, a, p):
	return y_0*x_s**p / (x_s**p + a**p)

def fun_min(args):
	y_0 = args[0]
	a = args[1]
	p = args[2]
	return (y_s - f(x_s, y_0, a, p)) / dy_s

res = least_squares(fun_min, [100, 20, 1], method='lm')
res_curve, M_cov = curve_fit(f, x_s, y_s, p0=[100, 20, 1], sigma=dy_s, absolute_sigma=True, method='lm')
print(np.sqrt(np.diag(M_cov)))
res_max = res.x - np.sqrt(np.diag(M_cov))*np.array([1, -1, 1])
res_min = res.x - np.sqrt(np.diag(M_cov))*np.array([-1, 1, -1])

fig, ax = plt.subplots()
M_corr = np.zeros_like(M_cov)
for i in range(len(M_corr)):
	for j in range(len(M_corr)):
		M_corr[i, j] = M_cov[i, j] / np.sqrt(M_cov[i, i]*M_cov[j,j])
im = ax.imshow(abs(M_corr), cmap='plasma', vmin=0)
ax.figure.colorbar(im)
lab = [r'$y_0$', 'a', 'p']
print(np.arange(3))
ax.set_xticks(np.arange(3))
ax.set_xticklabels(lab)
ax.set_yticks(np.arange(3))
ax.set_yticklabels(lab)
ax.xaxis.tick_top()

# Loop over data dimensions and create text annotations.
for i in range(len(M_corr)):
    for j in range(len(M_corr)):
        text = ax.text(j, i, round(M_corr[i, j], 2), ha="center", va="center", color="k")

fig.tight_layout()
plt.show()

'''
t_s = np.linspace(x_s[0],x_s[-1], 1000)
plt.plot(t_s, f(t_s, *res.x), alpha=0.6)
plt.plot(t_s, f(t_s, *res_max), 'k--', linewidth=1, alpha=0.3)
plt.plot(t_s, f(t_s, *res_min), 'k--', linewidth=1, alpha=0.3)
plt.fill_between(t_s, f(t_s, *res_max), f(t_s, *res_min), color='orange', alpha=0.2)
plt.errorbar(x_s, y_s, yerr=dy_s,fmt='.',capsize=3)
plt.xscale('log')
plt.show()'''