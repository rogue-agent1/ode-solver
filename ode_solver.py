#!/usr/bin/env python3
"""ODE solvers: Euler, RK4, adaptive RK45."""
def euler(f,y0,t0,t1,h):
    t,y=t0,y0;points=[(t,y)]
    while t<t1: y=y+h*f(t,y);t+=h;points.append((t,y))
    return points
def rk4(f,y0,t0,t1,h):
    t,y=t0,y0;points=[(t,y)]
    while t<t1:
        k1=f(t,y);k2=f(t+h/2,y+h/2*k1);k3=f(t+h/2,y+h/2*k2);k4=f(t+h,y+h*k3)
        y=y+h/6*(k1+2*k2+2*k3+k4);t+=h;points.append((t,y))
    return points
def rk45(f,y0,t0,t1,tol=1e-6,h=0.1):
    t,y=t0,y0;points=[(t,y)]
    while t<t1:
        k1=h*f(t,y);k2=h*f(t+h/4,y+k1/4);k3=h*f(t+3*h/8,y+3*k1/32+9*k2/32)
        k4=h*f(t+12*h/13,y+1932*k1/2197-7200*k2/2197+7296*k3/2197)
        k5=h*f(t+h,y+439*k1/216-8*k2+3680*k3/513-845*k4/4104)
        k6=h*f(t+h/2,y-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)
        y4=y+25*k1/216+1408*k3/2565+2197*k4/4104-k5/5
        y5=y+16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55
        err=abs(y5-y4)
        if err<tol: t+=h;y=y5;points.append((t,y))
        h*=0.9*(tol/max(err,1e-20))**0.2;h=min(h,t1-t) if t+h>t1 else h
    return points
if __name__=="__main__":
    import math
    f=lambda t,y:-y;exact=lambda t:math.exp(-t)
    e=euler(f,1.0,0,2,0.01);r=rk4(f,1.0,0,2,0.01);a=rk45(f,1.0,0,2)
    print(f"Euler err: {abs(e[-1][1]-exact(2)):.6f}")
    print(f"RK4 err: {abs(r[-1][1]-exact(2)):.10f}")
    print(f"RK45 err: {abs(a[-1][1]-exact(2)):.10f} ({len(a)} steps)")
    print("ODE solvers OK")
