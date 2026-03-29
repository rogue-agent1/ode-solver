#!/usr/bin/env python3
"""ode_solver - ODE integrators: Euler, RK4, adaptive RK45."""
import sys, json, math

def euler(f, y0, t0, t1, dt):
    t, y = t0, y0; history = [(t, y)]
    while t < t1:
        y = y + dt * f(t, y)
        t += dt; history.append((t, y))
    return history

def rk4(f, y0, t0, t1, dt):
    t, y = t0, y0; history = [(t, y)]
    while t < t1:
        k1 = f(t, y)
        k2 = f(t+dt/2, y+dt*k1/2)
        k3 = f(t+dt/2, y+dt*k2/2)
        k4 = f(t+dt, y+dt*k3)
        y = y + dt*(k1+2*k2+2*k3+k4)/6
        t += dt; history.append((t, y))
    return history

def rk45(f, y0, t0, t1, tol=1e-6, dt_init=0.1):
    t, y, dt = t0, y0, dt_init; history = [(t, y)]
    while t < t1:
        dt = min(dt, t1 - t)
        k1 = f(t, y)
        k2 = f(t+dt/4, y+dt*k1/4)
        k3 = f(t+3*dt/8, y+dt*(3*k1+9*k2)/32)
        k4 = f(t+12*dt/13, y+dt*(1932*k1-7200*k2+7296*k3)/2197)
        k5 = f(t+dt, y+dt*(439*k1/216-8*k2+3680*k3/513-845*k4/4104))
        k6 = f(t+dt/2, y+dt*(-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40))
        y4 = y+dt*(25*k1/216+1408*k3/2565+2197*k4/4104-k5/5)
        y5 = y+dt*(16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55)
        err = abs(y5-y4)
        if err < tol or dt < 1e-12:
            t += dt; y = y5; history.append((t, y))
            if err > 0: dt *= min(2, 0.9*(tol/err)**0.2)
        else:
            dt *= max(0.1, 0.9*(tol/err)**0.25)
    return history

def main():
    print("ODE solver demo\n")
    f = lambda t, y: -2*y + math.sin(t)
    exact = lambda t: (2*math.sin(t)-math.cos(t)+6*math.exp(-2*t))/5
    for name, solver in [("Euler", lambda: euler(f,1,0,3,0.1)), ("RK4", lambda: rk4(f,1,0,3,0.1)), ("RK45", lambda: rk45(f,1,0,3))]:
        h = solver()
        err = abs(h[-1][1] - exact(h[-1][0]))
        print(f"  {name:6s}: y(3)={h[-1][1]:.6f}, error={err:.2e}, steps={len(h)}")
    print(f"  Exact:  y(3)={exact(3):.6f}")

if __name__ == "__main__":
    main()
