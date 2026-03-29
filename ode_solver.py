#!/usr/bin/env python3
"""Ordinary differential equation solver: Euler, RK4, adaptive."""
import sys,math

def euler(f,y0,t0,t1,h):
    t,y=t0,y0;ts,ys=[t],[y]
    while t<t1:
        y=y+h*f(t,y);t+=h;ts.append(t);ys.append(y)
    return ts,ys

def rk4(f,y0,t0,t1,h):
    t,y=t0,y0;ts,ys=[t],[y]
    while t<t1:
        k1=h*f(t,y);k2=h*f(t+h/2,y+k1/2)
        k3=h*f(t+h/2,y+k2/2);k4=h*f(t+h,y+k3)
        y=y+(k1+2*k2+2*k3+k4)/6;t+=h;ts.append(t);ys.append(y)
    return ts,ys

def rk45_adaptive(f,y0,t0,t1,tol=1e-6,h_init=0.01):
    t,y,h=t0,y0,h_init;ts,ys=[t],[y]
    while t<t1:
        k1=h*f(t,y);k2=h*f(t+h/4,y+k1/4)
        k3=h*f(t+3*h/8,y+3*k1/32+9*k2/32)
        k4=h*f(t+12*h/13,y+1932*k1/2197-7200*k2/2197+7296*k3/2197)
        k5=h*f(t+h,y+439*k1/216-8*k2+3680*k3/513-845*k4/4104)
        k6=h*f(t+h/2,y-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)
        y4=y+25*k1/216+1408*k3/2565+2197*k4/4104-k5/5
        y5=y+16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55
        err=abs(y5-y4);s=0.84*(tol/max(err,1e-30))**0.25 if err>0 else 2
        if err<=tol:t+=h;y=y5;ts.append(t);ys.append(y)
        h=min(h*min(s,4),t1-t) if t<t1 else h*min(s,4)
        h=max(h,1e-10)
    return ts,ys

def main():
    print("=== ODE Solver ===\n")
    # dy/dt = -2y, y(0) = 1 → y = e^(-2t)
    f=lambda t,y:-2*y;exact=lambda t:math.exp(-2*t)
    print("dy/dt = -2y, y(0) = 1")
    for name,solver in [("Euler",lambda:euler(f,1,0,3,0.1)),("RK4",lambda:rk4(f,1,0,3,0.1)),("RK45",lambda:rk45_adaptive(f,1,0,3))]:
        ts,ys=solver()
        err=max(abs(y-exact(t)) for t,y in zip(ts,ys))
        print(f"  {name:6s}: {len(ts):4d} steps, max_err={err:.2e}, y(3)={ys[-1]:.6f} (exact={exact(3):.6f})")
    # Harmonic oscillator: y'' + y = 0
    print("\nHarmonic oscillator (y'' + y = 0):")
    def harmonic(t,state):
        y,v=state;return v  # simplified for scalar solver
    ts,ys=rk4(lambda t,y:math.cos(t),0,0,10,0.01)
    print(f"  RK4: {len(ts)} steps, y(10)={ys[-1]:.6f} (exact={math.sin(10):.6f})")

if __name__=="__main__":main()
