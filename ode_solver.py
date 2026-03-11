#!/usr/bin/env python3
"""ODE Solver — Euler, RK4, adaptive RK45 (Dormand-Prince)."""
import math, sys

def euler(f, y0, t_span, dt=0.01):
    t, y = t_span[0], y0; ts, ys = [t], [y]
    while t < t_span[1]:
        y = y + dt * f(t, y); t += dt
        ts.append(t); ys.append(y)
    return ts, ys

def rk4(f, y0, t_span, dt=0.01):
    t, y = t_span[0], y0; ts, ys = [t], [y]
    while t < t_span[1]:
        k1 = f(t, y); k2 = f(t+dt/2, y+dt*k1/2)
        k3 = f(t+dt/2, y+dt*k2/2); k4 = f(t+dt, y+dt*k3)
        y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6; t += dt
        ts.append(t); ys.append(y)
    return ts, ys

def rk45(f, y0, t_span, tol=1e-6, dt=0.01):
    """Adaptive RK45 (Dormand-Prince coefficients)."""
    a2,a3,a4,a5,a6 = 1/4, 3/8, 12/13, 1, 1/2
    b = [[],[1/4],[3/32,9/32],[1932/2197,-7200/2197,7296/2197],
         [439/216,-8,3680/513,-845/4104],[-8/27,2,-3544/2565,1859/4104,-11/40]]
    c = [25/216,0,1408/2565,2197/4104,-1/5,0]
    cs = [16/135,0,6656/12825,28561/56430,-9/50,2/55]
    t, y = t_span[0], y0; ts, ys = [t], [y]
    while t < t_span[1]:
        k = [f(t, y)]
        for i in range(1, 6):
            ti = t + [0,a2,a3,a4,a5,a6][i]*dt
            yi = y + dt*sum(b[i][j]*k[j] for j in range(i))
            k.append(f(ti, yi))
        y4 = y + dt*sum(c[i]*k[i] for i in range(6))
        y5 = y + dt*sum(cs[i]*k[i] for i in range(6))
        err = abs(y5 - y4)
        if err < tol or dt < 1e-12:
            y = y5; t += dt; ts.append(t); ys.append(y)
        dt = min(dt * max(0.1, 0.84*(tol/max(err,1e-30))**0.25), t_span[1]-t) if err > 0 else dt
    return ts, ys

if __name__ == "__main__":
    f = lambda t, y: -2*y  # dy/dt = -2y, exact: e^(-2t)
    exact = lambda t: math.exp(-2*t)
    for name, solver in [("Euler", euler), ("RK4", rk4)]:
        ts, ys = solver(f, 1.0, (0, 2), dt=0.1)
        err = abs(ys[-1] - exact(ts[-1]))
        print(f"{name:6s}: y(2)={ys[-1]:.8f}, error={err:.2e}")
    ts, ys = rk45(f, 1.0, (0, 2))
    print(f"RK45:   y(2)={ys[-1]:.8f}, error={abs(ys[-1]-exact(ts[-1])):.2e}, steps={len(ts)}")
