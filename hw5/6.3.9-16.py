import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t, z):
    x, y = z
    return [y**3 - 4*x, y**3 - y - 3*x]

#domain
L = 20

n = 50
xs = np.linspace(-L, L, n)
ys = np.linspace(-L, L, n)
X, Y = np.meshgrid(xs, ys)

U = Y**3 - 4*X
V = Y**3 - Y - 3*X
S = np.sqrt(U**2 + V**2) + 1e-12
U_n, V_n = U/S, V/S

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, U_n, V_n, alpha=0.55, scale=45)

#important stuff
line = np.linspace(-L, L, 600)
ax.plot(line, line, 'm-', lw=2, label='x=y')

# fixed points
ax.plot([0, 2, -2], [0, 2, -2], 'ro', ms=6, label='fixed points')


def plot_traj(z0, T=8.0, dt=0.01):
    # ft
    sol_f = solve_ivp(
        f, (0, T), z0,
        rtol=1e-8, atol=1e-10,
        max_step=dt
    )
    ax.plot(sol_f.y[0], sol_f.y[1], lw=1.2)

    # bt
    sol_b = solve_ivp(
        f, (0, -T), z0,
        rtol=1e-8, atol=1e-10,
        max_step=dt
    )
    ax.plot(sol_b.y[0], sol_b.y[1], lw=1.0, ls=':')

#initial conditions
inits = []
for x0 in [-15, -8, 0, 8, 15]:
    for y0 in [-15, -8, 0, 8, 15]:
        if (x0, y0) != (0, 0):
            inits.append((x0, y0))

for z0 in inits:
    plot_traj(z0, T=6.0, dt=0.01)

#plot:)
ax.axhline(0, lw=1)
ax.axvline(0, lw=1)
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect('equal', 'box')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('6.3.9 d')
ax.legend(loc='upper left', fontsize='small')
plt.show()


### 16

def phase_portrait(a, xlim=(-3, 3), ylim=(-3, 3), ngrid=400):
    x = np.linspace(*xlim, ngrid)
    y = np.linspace(*ylim, ngrid)
    X, Y = np.meshgrid(x, y)

    U = a + X**2 - X*Y
    V = Y**2 - X**2 - 1

    speed = np.hypot(U, V)
    U2 = U / (speed + 1e-12)
    V2 = V / (speed + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.streamplot(X, Y, U2, V2, density=1.2, linewidth=0.8, arrowsize=1)

    xx = np.linspace(*xlim, 2000)
    yplus = np.sqrt(xx**2 + 1)
    yminus = -np.sqrt(xx**2 + 1)
    ax.plot(xx, yplus, 'k--', lw=1, label=r'$\dot y=0$')
    ax.plot(xx, yminus, 'k--', lw=1)

    eps = 1e-3
    xx1 = np.linspace(xlim[0], -eps, 2000)
    xx2 = np.linspace(eps, xlim[1], 2000)
    ax.plot(xx1, xx1 + a/xx1, 'r--', lw=1, label=r'$\dot x=0$')
    ax.plot(xx2, xx2 + a/xx2, 'r--', lw=1)

    if 1 - 2*a > 0:
        xeq_mag = abs(a) / np.sqrt(1 - 2*a)
        xeq = np.array([xeq_mag, -xeq_mag])
        yeq = xeq + a/xeq
        ax.plot(xeq, yeq, 'bo', ms=6, label='fixed points')

    ax.set_title(f"a = {a}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.legend(loc="upper left")
    ax.set_aspect('equal', adjustable='box')
    plt.show()

phase_portrait(a=0)
phase_portrait(a=-0.5)
phase_portrait(a=+0.5)
