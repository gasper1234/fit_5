import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

datContent = [i.strip().split() for i in open("ledvice.dat").readlines()]
t_s = []
N_s = []

for line in datContent[1:]:
	t_s.append(float(line[0]))
	N_s.append(float(line[1]))

t_s = np.array(t_s)
N_s = np.array(N_s)

def f1(t, N, lamb):
	return N*np.exp(-lamb*t)

#plt.plot(t_s, f1(t_s, 13000, 0.002), ':')


def f2(t, N, lamb, A):
	return N*np.exp(-lamb*t) + A

def f3(t, N_1, lamb_1, N_2, lamb_2, A):
	return N_1*np.exp(-lamb_1*t) + N_2*np.exp(-lamb_2*t) + A

def f4(t, N, lamb, A):
	return N*np.exp(-lamb*np.sqrt(t)) + A

f_s = [f1, f2, f3, f4]
init_s = [[13000, 0.002], [13000, 0.002, 2000], [7000, 0.002, 7000, 0.002, 2000], [13000, 0.002, 2000]]
t_span = np.linspace(0, t_s[-1], 1000)
lab_s = [['N', r'$\lambda$'], ['N', r'$\lambda$', 'A'], [r'$N_1$', r'$\lambda_1$', r'$N_2$', r'$\lambda_2$', 'A'], ['N', r'$\lambda$', 'A']]
res_arg_max = np.array([[1, -1], [1, -1, 1], [1, -1, 1, -1, 1], [1, -1, 1]])
res_arg_min = np.array([[j*(-1) for j in res_arg_max[i]] for i in range(len(res_arg_max))])
label_s = [r'$N\mathrm{e}^{-\lambda t}$', r'$N\mathrm{e}^{-\lambda t}+A$', r'$N_1\mathrm{e}^{-\lambda_1 t}+N_2\mathrm{e}^{-\lambda_2 t}+A$', r'$N\mathrm{e}^{-\lambda \sqrt{t}}$']
fig, ax = plt.subplots(4,2)
fig.set_figheight(24)
fig.set_figwidth(10)

def chi_square(f, y):
	return np.sum((f-y)**2)

for i in range(len(f_s)):
	fit_params, fit_cov = curve_fit(f_s[i], t_s, N_s, p0=init_s[i])
	ax[i][0].plot(t_s, N_s, 'rx', label='data')
	ax[i][0].plot(t_span, f_s[i](t_span, *fit_params), label=label_s[i])
	ax[i][0].plot(t_span, f_s[i](t_span, *(fit_params+np.sqrt(np.diag(fit_cov))*res_arg_max[i])), 'k--', alpha=0.3)
	ax[i][0].plot(t_span, f_s[i](t_span, *(fit_params+np.sqrt(np.diag(fit_cov))*res_arg_min[i])), 'k--', alpha=0.3)
	ax[i][0].fill_between(t_span, f_s[i](t_span, *(fit_params+np.sqrt(np.diag(fit_cov))*res_arg_max[i])), f_s[i](t_span, *(fit_params+np.sqrt(np.diag(fit_cov))*res_arg_min[i])), color='orange', alpha=0.2)
	print(fit_cov)
	ax[i][0].set_ylabel('N')
	ax[i][0].legend(title=r'$\chi^2\cdot 10^{-3}=$'+str(int(round(chi_square(f_s[i](t_s, *fit_params), N_s)/1000))))
	M_corr = np.zeros_like(fit_cov)
	for k in range(len(M_corr)):
		for j in range(len(M_corr)):
			M_corr[k, j] = fit_cov[k, j] / np.sqrt(fit_cov[k, k]*fit_cov[j,j])
	im = ax[i][1].imshow(abs(M_corr), cmap='plasma', vmin=0)
	for k in range(len(M_corr)):
		for j in range(len(M_corr)):
			text = ax[i][1].text(j, k, round(M_corr[k, j], 2), ha="center", va="center", color="k")
	fig.colorbar(im, ax=ax[i][1])
	lab = lab_s[i]
	ax[i][1].set_xticks(np.arange(len(M_corr)))
	ax[i][1].set_xticklabels(lab)
	ax[i][1].set_yticks(np.arange(len(M_corr)))
	ax[i][1].set_yticklabels(lab)
	ax[i][1].xaxis.tick_top()


ax[3][0].set_xlabel('t')
plt.show()

#dodaj še fitane funkcije in morda rezultate fita?