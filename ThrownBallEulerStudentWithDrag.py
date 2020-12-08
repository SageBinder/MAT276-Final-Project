# Written by Chris McCarthy July 2019 SIMIODE DEMARC
# Drag = 0 Student Version 
#===========================================  for online compilers
import matplotlib as mpl
#===========================================  usual packages
import numpy as np
from matplotlib import pyplot as plt
import math
plt.rcParams.update({'font.size': 12})

class Ball:
    def __init__(self, x, y, vx, vy, t, m, c):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.t = t

        self.m = m
        self.c = c

    def update_ball(self, delta_t, g, drag=True):
        self.x = self.x + delta_t * self.vx
        self.y = self.y + delta_t * self.vy

        if drag:
            v = np.sqrt(self.vx ** 2 + self.vy ** 2)
            Fd = -self.c * (v * v)
            cos_theta = self.vx / v
            sin_theta = self.vy / v
            Fd_x = Fd * cos_theta
            Fd_y = Fd * sin_theta
        else:
            Fd_x = 0
            Fd_y = 0

        Fg = -g * self.m

        self.vx = self.vx + delta_t * (Fd_x / self.m)
        self.vy = self.vy + delta_t * ((Fg + Fd_y) / self.m)
        self.t = self.t + delta_t

def trajectory(ball, g, dt):
    xvalues = []
    yvalues = []

    while 0 <= ball.y:
        ball.update_ball(dt, g, drag=drag)
        xvalues.append(ball.x)
        yvalues.append(ball.y)

    return (xvalues, yvalues)

def plot_trajectory(ball, g, dt, drag=True, label=None, style='-', color=None):
    xvalues = []
    yvalues = []

    while 0 <= ball.y:
        ball.update_ball(dt, g, drag=drag)
        xvalues.append(ball.x)
        yvalues.append(ball.y)

    plt.plot(xvalues, yvalues, style, label=label, color=color)

def plot_best_ball(x0, y0, speed, dt, get_ball, g, label=None, drag=True):
    plot_trajectory(find_best_ball(x0, y0, speed, dt, get_ball, drag), g, dt, label, drag)

def find_best_ball(x0, y0, speed, dt, get_ball, drag=True):
    #============================================ Delta t
    xDistance = []   # store the horizontal distance ball travelled
    Theta = []       # store the angle the ball is thrown at
    #============================================ run Euler for theta in [0, 90]
    for theta in range(0, 90):
        theta_rads = np.radians(theta)
        ball = get_ball(x0, y0, t0, speed, theta_rads)   # initialize ball object
        while 0 <= ball.y:                    # Euler Method applied to that ball
            ball.update_ball(dt, g, drag=drag)
        xDistance.append(ball.x)             # collect x value when ball hits ground 
        Theta.append(theta_rads)                  # collect theta 
    #============================================ find max x distance over theta, print it
    maxpos = xDistance.index(max(xDistance))
    #============================================= run Euler (again) for best theta
    best_theta = Theta[maxpos]
    best_vx0 = speed*np.cos(best_theta) # initial vx
    best_vy0 = speed*np.sin(best_theta) # initial vy
    best_ball = get_ball(x0, y0, t0, speed, best_theta)   # initialize ball object

    return best_ball

def find_best_ball_and_theta(x0, y0, speed, dt, get_ball, drag=True):
    #============================================ Delta t
    xDistance = []   # store the horizontal distance ball travelled
    Theta = []       # store the angle the ball is thrown at
    #============================================ run Euler for theta in [0, 90]
    for theta in range(0, 90):
        theta_rads = np.radians(theta)
        ball = get_ball(x0, y0, t0, speed, theta_rads)   # initialize ball object
        while 0 <= ball.y:                    # Euler Method applied to that ball
            ball.update_ball(dt, g, drag=drag)
        xDistance.append(ball.x)             # collect x value when ball hits ground 
        Theta.append(theta_rads)                  # collect theta 
    #============================================ find max x distance over theta, print it
    maxpos = xDistance.index(max(xDistance))
    #============================================= run Euler (again) for best theta
    best_theta = Theta[maxpos]
    best_vx0 = speed*np.cos(best_theta) # initial vx
    best_vy0 = speed*np.sin(best_theta) # initial vy
    best_ball = get_ball(x0, y0, t0, speed, best_theta)   # initialize ball object

    return (best_ball, best_theta)

d = 1.21    # density of air

def create_golf_ball(x0, y0, t0, speed, theta):
    # Golf ball constants
    m = 0.046   # mass of ball
    r = 0.0213  # radius of ball
    C_d = 0.5   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

def create_tennis_ball(x0, y0, t0, speed, theta):
    # Tennis ball constants
    m = 0.059   # mass of ball
    r = 0.0327  # radius of ball
    C_d = 0.6   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

def create_soccer_ball(x0, y0, t0, speed, theta):
    # Soccer ball constants
    m = 0.454   # mass of ball
    r = 0.11  # radius of ball
    C_d = 0.5   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

def create_racquetball(x0, y0, t0, speed, theta):
    # Racquetball constants
    m = 0.04   # mass of ball
    r = 0.026  # radius of ball
    C_d = 0.5   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

def create_ping_pong_ball(x0, y0, t0, speed, theta):
    # Ping pong ball constants
    m = 0.0027   # mass of ball
    r = 0.02  # radius of ball
    C_d = 0.5   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

def create_wiffle_ball(x0, y0, t0, speed, theta):
    # Wiffle ball constants
    m = 0.02   # mass of ball
    r = 0.04  # radius of ball
    C_d = 1   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

def create_shuttlecock(x0, y0, t0, speed, theta):
    # Shuttlecock constants
    m = 0.0055   # mass of ball
    r = 0.068  # radius of ball
    C_d = 0.6   # coefficient of drag
    c = (C_d * d * np.pi * r * r) / 2

    return Ball(x0, y0, speed * np.cos(theta), speed * np.sin(theta), t0, m, c)

g = 9.8     # gravitational acceleration
x0 = 0      # initial x position in meters
y0 = 2      # initial y position in meters
t0 = 0      # initial time in seconds
speed = 12  # initial speed of the ball in meters/sec
dt = .001   # Delta t

dpi = 110
xpixels = 1800
ypixels = 900

def make_fig():
    fig = plt.figure(figsize=(xpixels/dpi, ypixels/dpi), dpi=dpi)
    plt.xlabel("Distance [m]")
    plt.ylabel("Height [m]")

    return fig

def grid():
    plt.grid(linewidth='3', color='black')
    plt.gca().set_aspect('equal', adjustable='box')

def plot_all_balls_at_45(show_fig=True):
    fig = make_fig()

    plot_trajectory(create_tennis_ball(x0, y0, t0, 12, np.pi / 4), g, dt, label="Tennis ball")
    plot_trajectory(create_ping_pong_ball(x0, y0, t0, 12, np.pi / 4), g, dt, label="Ping pong ball")
    plot_trajectory(create_golf_ball(x0, y0, t0, 12, np.pi / 4), g, dt, label="Golf ball")
    plot_trajectory(create_soccer_ball(x0, y0, t0, 12, np.pi / 4), g, dt, label="Soccer ball")

    grid()
    plt.legend(loc='upper right')
    plt.title("Different balls thrown at 45 degrees, initial speed of 12 m/s")

    if show_fig: 
        plt.show()

    return fig

def plot_all_balls_at_ideal(show_fig=True):
    fig = make_fig()

    plot_trajectory(find_best_ball(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_tennis_ball(x0, y0, t0, speed, theta)), g, dt, label="Tennis ball")
    plot_trajectory(find_best_ball(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_ping_pong_ball(x0, y0, t0, speed, theta)), g, dt, label="Ping pong ball")
    plot_trajectory(find_best_ball(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_golf_ball(x0, y0, t0, speed, theta)), g, dt, label="Golf ball")
    plot_trajectory(find_best_ball(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_soccer_ball(x0, y0, t0, speed, theta)), g, dt, label="Soccer ball")

    grid()
    plt.legend(loc='upper right')
    plt.title("Different balls thrown at ideal angles, initial speed of 12 m/s")

    if show_fig: 
        plt.show()

    return fig

def plot_no_drag_ball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_golf_ball(x0, y0, t0, speed, theta), drag=False)
    plot_trajectory(best_ball, g, dt, label=f"Without drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=False)
    plot_trajectory(create_golf_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="Without drag, 45 degrees", color=color, style='--', drag=False)

def plot_golf_ball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_golf_ball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Golf ball with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_golf_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="Golf ball with drag, 45 degrees", color=color, style='--', drag=True)

def plot_racquetball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_racquetball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Racquetball with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_racquetball(x0, y0, t0, speed, np.pi/4), g, dt, label="Racquetball with drag, 45 degrees", color=color, style='--', drag=True)

def plot_tennis_ball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_tennis_ball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Tennis ball with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_tennis_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="Tennis ball with drag, 45 degrees", color=color, style='--', drag=True)

def plot_soccer_ball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_soccer_ball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Soccer ball with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_soccer_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="Soccer ball with drag, 45 degrees", color=color, style='--', drag=True)

def plot_ping_pong_ball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_ping_pong_ball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Ping pong ball with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_ping_pong_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="Ping pong ball with drag, 45 degrees", color=color, style='--', drag=True)

def plot_wiffle_ball(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_wiffle_ball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Wiffle ball with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_wiffle_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="Wiffle ball with drag, 45 degrees", color=color, style='--', drag=True)

def plot_shuttlecock(color):
    best_ball, best_theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_shuttlecock(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Shuttlecock with drag, ideal angle ({round(np.degrees(best_theta))} degrees)", color=color, drag=True)
    plot_trajectory(create_shuttlecock(x0, y0, t0, speed, np.pi/4), g, dt, label="Shuttlecock with drag, 45 degrees", color=color, style='--', drag=True)

def drag_comparison(show_fig=True):
    fig = make_fig()

    plot_no_drag_ball('blue')
    plot_golf_ball('orange')
    plot_racquetball('brown')
    plot_tennis_ball('green')
    plot_soccer_ball('yellow')
    plot_ping_pong_ball('red')
    plot_wiffle_ball('pink')
    plot_shuttlecock('purple')

    grid()
    plt.legend(loc='upper right', prop={'size': 7})
    plt.title(f"Comparing each ball with drag to ideal no-drag trajectory, initial speed of {speed} m/s")

    if show_fig: 
        plt.show()

    return fig

def tennis_ball_at_multiple_angles(show_fig=True):
    plot_trajectory(create_tennis_ball(x0, y0, t0, speed, 0), g, dt, label="0 degrees", style='--', drag=True)
    plot_trajectory(create_tennis_ball(x0, y0, t0, speed, np.pi/8), g, dt, label="22.25 degrees", style='--', drag=True)
    plot_trajectory(create_tennis_ball(x0, y0, t0, speed, np.pi/4), g, dt, label="45 degrees", style='--', drag=True)

    best_ball, theta = find_best_ball_and_theta(x0, y0, speed, dt, lambda x0, y0, t0, speed, theta: create_tennis_ball(x0, y0, t0, speed, theta), drag=True)
    plot_trajectory(best_ball, g, dt, label=f"Ideal angle ({np.degrees(theta)} degrees)", drag=True)


# fig = plot_all_balls_at_45(show_fig=True)
# fig.savefig("All balls at 45 degrees.pdf", dpi=dpi)

# fig = plot_all_balls_at_ideal(show_fig=True)
# fig.savefig("All balls at ideal angle.pdf", dpi=dpi)

# fig = drag_comparison()
# fig.savefig("Drag comparison.pdf")

# SHUTTLECOCK VS GOLF BALL VS NO DRAG
make_fig()
plot_no_drag_ball('blue')
plot_golf_ball('orange')
plot_shuttlecock('purple')
grid()
plt.legend(loc='upper right', prop={'size': 8})
plt.title(f"Shuttlecock vs. Golf Ball vs. No Drag, initial speed of {speed} m/s")
plt.savefig("Shuttlecock vs. Golf Ball vs. No Drag.pdf")
plt.show()


# TENNIS BALL AT MULTIPLE ANGLES, AND AT IDEAL ANGLE
# make_fig()
# tennis_ball_at_multiple_angles()
# grid()
# plt.legend(loc='upper right', prop={'size': 8})
# plt.title(f"Tennis ball with drag, multiple angles, initial speed of {speed} m/s")
# plt.savefig("Tennis ball at multiple angles.pdf")
# plt.show()