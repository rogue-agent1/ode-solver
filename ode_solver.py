#!/usr/bin/env python3
"""ode_solver - ODE solver: Euler, RK4, adaptive RK45."""
import sys, argparse, json, math

def euler(f, y0, t0, t1, dt):
    t, y = t0, list(y0); history = [{"t": t, "y": list(y)}]
    while t < t1:
        dy = f(t, y)
        y = [yi + dyi * dt for yi, dyi in zip(y, dy)]
        t += dt
        history.append({"t": round(t, 8), "y": [round(v, 8) for v in y]})
    return history

def rk4(f, y0, t0, t1, dt):
    t, y = t0, list(y0); history = [{"t": t, "y": list(y)}]
    while t < t1:
        k1 = f(t, y)
        k2 = f(t + dt/2, [yi + ki*dt/2 for yi, ki in zip(y, k1)])
        k3 = f(t + dt/2, [yi + ki*dt/2 for yi, ki in zip(y, k2)])
        k4 = f(t + dt, [yi + ki*dt for yi, ki in zip(y, k3)])
        y = [yi + dt/6*(k1i + 2*k2i + 2*k3i + k4i) for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)]
        t += dt
        history.append({"t": round(t, 8), "y": [round(v, 8) for v in y]})
    return history

def rk45(f, y0, t0, t1, tol=1e-6, dt_init=0.01):
    t, y, dt = t0, list(y0), dt_init; history = [{"t": t, "y": list(y)}]
    while t < t1:
        dt = min(dt, t1 - t)
        k1 = f(t, y)
        k2 = f(t+dt/4, [yi+dt/4*ki for yi,ki in zip(y,k1)])
        k3 = f(t+3*dt/8, [yi+dt*(3*k1i/32+9*k2i/32) for yi,k1i,k2i in zip(y,k1,k2)])
        k4 = f(t+12*dt/13, [yi+dt*(1932*k1i/2197-7200*k2i/2197+7296*k3i/2197) for yi,k1i,k2i,k3i in zip(y,k1,k2,k3)])
        y4 = [yi+dt*(439*k1i/216-8*k2i+3680*k3i/513-845*k4i/4104) for yi,k1i,k2i,k3i,k4i in zip(y,k1,k2,k3,k4)]
        k5 = f(t+dt, y4)
        k6 = f(t+dt/2, [yi+dt*(-8*k1i/27+2*k2i-3544*k3i/2565+1859*k4i/4104-11*k5i/40) for yi,k1i,k2i,k3i,k4i,k5i in zip(y,k1,k2,k3,k4,k5)])
        y5 = [yi+dt*(16*k1i/135+6656*k3i/12825+28561*k4i/56430-9*k5i/50+2*k6i/55) for yi,k1i,k3i,k4i,k5i,k6i in zip(y,k1,k3,k4,k5,k6)]
        err = max(abs(a-b) for a,b in zip(y4,y5)) or 1e-16
        if err <= tol:
            t += dt; y = y5
            history.append({"t": round(t,8), "y": [round(v,8) for v in y]})
        dt *= min(2, max(0.1, 0.84*(tol/err)**0.25))
    return history

def main():
    p = argparse.ArgumentParser(description="ODE solver")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        exp_decay = lambda t, y: [-0.5 * y[0]]
        print("=== Exponential Decay y'=-0.5y ===")
        for name, solver in [("Euler", euler), ("RK4", rk4)]:
            h = solver(exp_decay, [1.0], 0, 5, 0.1)
            exact = math.exp(-0.5*5)
            print(f"{name}: final={h[-1]['y'][0]:.6f} exact={exact:.6f} err={abs(h[-1]['y'][0]-exact):.2e}")
        h = rk45(exp_decay, [1.0], 0, 5)
        print(f"RK45: final={h[-1]['y'][0]:.6f} steps={len(h)}")
        print("\n=== Lorenz System ===")
        def lorenz(t, y):
            s, r, b = 10, 28, 8/3
            return [s*(y[1]-y[0]), y[0]*(r-y[2])-y[1], y[0]*y[1]-b*y[2]]
        h = rk4(lorenz, [1,1,1], 0, 10, 0.01)
        print(f"Lorenz at t=10: {h[-1]['y']}")
    else: p.print_help()

if __name__ == "__main__":
    main()
