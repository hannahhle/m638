import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# og system
def f(t, z):
    x, y = z
    return [x + np.exp(-y), -y]

# fixed point
xfp = np.array([-1.0, 0.0])

# jacobian at equilibrium
J = np.array([[1, -1],
              [0, -1]])

eigvals, eigvecs = np.linalg.eig(J)

# stable eigenvector
idx = np.argmin(np.real(eigvals)) 
v_stable = np.real(eigvecs[:, idx])
v_stable = v_stable / np.linalg.norm(v_stable)

# stable manifold (hopefully lol) 
def y_series(x):
    u = x + 1.0
    return 2.0*u + (4.0/3.0)*u**2

fig, ax = plt.subplots(figsize=(8,6))

# vector field
X, Y = np.meshgrid(np.linspace(-2, 1, 30), np.linspace(-1, 1, 30))
U = X + np.exp(-Y)
V = -Y
speed = np.sqrt(U**2 + V**2)
ax.quiver(X, Y, U/speed, V/speed, scale=30, alpha=0.6)

#fixed point
ax.plot(xfp[0], xfp[1], 'o', color='orange', ms=4, label='fixed point (-1,0)')

# stable manifold i hope this works
eps = 1e-4
Tback = 8.0
t_span = (0.0, -Tback)   # integrate backward
t_eval = np.linspace(0.0, -Tback, 1000)

for sign in [+1, -1]:
    z0 = xfp + sign*eps*v_stable
    sol = solve_ivp(f, t_span, z0, t_eval=t_eval, atol=1e-10, rtol=1e-9)
    ax.plot(sol.y[0], sol.y[1], 'k-', lw=2, label='stable manifold' if sign==1 else None)

init_points = [(-0.5, 0.8), (-0.7, 0.4), (-1.3, -0.2)]
t_fwd = (0.0, 8.0)
t_eval_fwd = np.linspace(0, 8, 400)
for z0 in init_points:
    sol = solve_ivp(f, t_fwd, z0, t_eval=t_eval_fwd)
    ax.plot(sol.y[0], sol.y[1], lw=1)

ax.set_xlim(-2, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('6.1.14')
ax.legend(loc='lower right', fontsize='small')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
plt.show()