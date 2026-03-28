#!/usr/bin/env python3
"""ode_solver - Ordinary differential equation solvers."""
import argparse, math, json

def euler(f, y0, t0, t_end, dt):
    t, y = t0, y0; history = [(t, y)]
    while t < t_end:
        y = y + dt * f(t, y); t += dt
        history.append((t, y))
    return history

def rk4(f, y0, t0, t_end, dt):
    t, y = t0, y0; history = [(t, y)]
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t+dt/2, y+dt/2*k1)
        k3 = f(t+dt/2, y+dt/2*k2)
        k4 = f(t+dt, y+dt*k3)
        y = y + dt/6*(k1+2*k2+2*k3+k4); t += dt
        history.append((t, y))
    return history

def euler_system(f, y0, t0, t_end, dt):
    t, y = t0, list(y0); history = [(t, list(y))]
    while t < t_end:
        dy = f(t, y)
        y = [yi + dt*dyi for yi, dyi in zip(y, dy)]; t += dt
        history.append((t, list(y)))
    return history

def rk4_system(f, y0, t0, t_end, dt):
    t = t0; y = list(y0); n = len(y); history = [(t, list(y))]
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t+dt/2, [y[i]+dt/2*k1[i] for i in range(n)])
        k3 = f(t+dt/2, [y[i]+dt/2*k2[i] for i in range(n)])
        k4 = f(t+dt, [y[i]+dt*k3[i] for i in range(n)])
        y = [y[i]+dt/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]) for i in range(n)]; t += dt
        history.append((t, list(y)))
    return history

def main():
    p = argparse.ArgumentParser(description="ODE solver")
    p.add_argument("--demo", choices=["exponential", "harmonic", "lorenz", "predator-prey"], default="exponential")
    p.add_argument("-m", "--method", choices=["euler", "rk4"], default="rk4")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--t-end", type=float, default=10)
    args = p.parse_args()
    if args.demo == "exponential":
        f = lambda t, y: -0.5 * y
        history = rk4(f, 1.0, 0, args.t_end, args.dt) if args.method=="rk4" else euler(f, 1.0, 0, args.t_end, args.dt)
        exact = lambda t: math.exp(-0.5*t)
        for t, y in history[::len(history)//10]:
            print(f"  t={t:.2f}: y={y:.6f} exact={exact(t):.6f} err={abs(y-exact(t)):.2e}")
    elif args.demo == "harmonic":
        f = lambda t, y: [y[1], -y[0]]
        history = rk4_system(f, [1.0, 0.0], 0, args.t_end, args.dt)
        for t, y in history[::len(history)//10]:
            print(f"  t={t:.2f}: x={y[0]:.4f} v={y[1]:.4f}")
    elif args.demo == "lorenz":
        sigma, rho, beta = 10, 28, 8/3
        f = lambda t, y: [sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]]
        history = rk4_system(f, [1,1,1], 0, min(args.t_end, 50), args.dt)
        print(f"Lorenz attractor ({len(history)} points)")
        for t, y in history[::len(history)//20]:
            print(f"  t={t:.1f}: ({y[0]:.2f}, {y[1]:.2f}, {y[2]:.2f})")
    elif args.demo == "predator-prey":
        a, b, c, d = 1.1, 0.4, 0.4, 0.1
        f = lambda t, y: [a*y[0]-b*y[0]*y[1], -c*y[1]+d*y[0]*y[1]]
        history = rk4_system(f, [10, 5], 0, args.t_end, args.dt)
        for t, y in history[::len(history)//10]:
            print(f"  t={t:.1f}: prey={y[0]:.1f} predator={y[1]:.1f}")

if __name__ == "__main__":
    main()
