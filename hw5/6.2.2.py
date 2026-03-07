import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# system def
def f(t, z):
    x, y = z
    dx = y
    dy = -x + (1 - x**2 - y**2) * y
    return [dx, dy]

# phase portrait
xlim = (-2.5, 2.5)
ylim = (-2.5, 2.5)

X, Y = np.meshgrid(np.linspace(*xlim, 35), np.linspace(*ylim, 35))
U = Y
V = -X + (1 - X**2 - Y**2) * Y

# normalize for nicer arrows
S = np.sqrt(U**2 + V**2) + 1e-12
U_n, V_n = U / S, V / S

fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(X, Y, U_n, V_n, scale=30, alpha=0.6)

# axes lines
ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

### part b
t = np.linspace(0, 2*np.pi, 600)
xb = np.sin(t)
yb = np.cos(t)

### part c
z0 = [0.5, 0.0]
t_span = (0, 50)
t_eval = np.linspace(*t_span, 4000)

sol = solve_ivp(f, t_span, z0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
xc, yc = sol.y
ax.plot(xc, yc, 'b', linewidth=2, label='orbit from (1/2, 0)')
ax.plot(xb, yb, 'r--',linewidth=2, label='(sin t, cos t)')


# plot the start point
ax.plot(z0[0], z0[1], 'bo', markersize=3)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect('equal', 'box')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('6.2.2 c')
ax.legend(loc='upper right')
plt.show()

