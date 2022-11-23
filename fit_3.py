import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd as svd


datContent = [i.strip().split() for i in open("3.dat").readlines()]
ttg_s = []
xfp_s = []
tfp_s = []

for line in datContent:
	ttg_s.append(float(line[0]))
	xfp_s.append(float(line[1]))
	tfp_s.append(float(line[2]))

ttg_s = np.array(ttg_s)
xfp_s = np.array(xfp_s)
tfp_s = np.array(tfp_s)

def make_A(p, rem):
	print(100000000)
	A = np.zeros((len(ttg_s), int((p+1)*p/2)-len(rem)))
	for ind in range(len(ttg_s)):
		x, theta = xfp_s[ind], ttg_s[ind]
		line_ind = 0
		for i in range(p):
			for j in range(p):
				if (j+i) < p:
					if (i, j) in rem:
						continue
					A[ind, line_ind] = (x**i * theta**j)
					line_ind += 1
					if ind == 1:
						print('x_pot='+str(i), 'theta='+str(j))
	return A

def a_sol(U, s, V, b):
	a = np.zeros(len(V[0]))
	for i in range(len(V)):
		a += np.dot(U[:, i], b) / s[i] * V[i]
	return a

def b_sol(a, p, rem):
	b = np.zeros(len(ttg_s))
	for ind in range(len(ttg_s)):
		x, theta = xfp_s[ind], ttg_s[ind]
		line_ind = 0
		for i in range(p):
			for j in range(p):
				if (j+i) < p:
					if (i, j) in rem:
						continue
					b[ind] += (x**i * theta**j)*a[line_ind]
					line_ind += 1
	return b

def err_vec(V, s):
	a_err = np.zeros(len(V))
	for j in range(len(V[0])):
		for i in range(len(V)):
			a_err[j] += V[i, j] / s[i]**2
	return a_err

def chi_sq(b, b_sol):
	chi = 0
	for i in range(len(b)):
		chi += (b[i]-b_sol[i])**2		
	return chi

def cov_mat(V, s):
	V_cov = np.zeros_like(V)
	for i in range(len(V)):
		for j in range(len(V[0])):
			for k in range(len(V)):
				V_cov[i, j] += V[k, i]*V[k, j] / s[k]**2
	V_1 = np.copy(V_cov)
	for i in range(len(V)):
		for j in range(len(V)):
			V_1[i, j] /= np.sqrt(V_cov[j, j]*V_cov[i, i])
	return V_1

for p in range(5, 6):
	rem = [(4, 0), (3,0)]
	#rem = []
	A = make_A(p, rem)

	U, s, V = svd(A, full_matrices=True)
	#print(s)
	#print(np.diag(V))
	a = a_sol(U, s, V, tfp_s)
	a_err = err_vec(V, s)
	#print(a)
	#print(a_err)


	b = b_sol(a, p, rem)
	uspesnost = chi_sq(b, tfp_s)
	fig, ax = plt.subplots()
	M_corr = cov_mat(V, s)
	im = ax.imshow(abs(M_corr), cmap='plasma', vmin=0)
	ax.figure.colorbar(im)
	ax.set_title(r'$\chi^2=$'+str(round(uspesnost)))
	for i in range(len(M_corr)):
		for j in range(len(M_corr)):
			text = ax.text(j, i, round(M_corr[i, j], 2), ha="center", va="center", color="k")
	ax.xaxis.tick_top()
	plt.show()

'''
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	ax.scatter(xfp_s, ttg_s, tfp_s, color='red')
	ax.scatter(xfp_s, ttg_s, b)
	ax.set_xlabel('x')
	plt.show()
'''