from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec



def pendulum_euler(initial_conditions=(2, np.pi/2, 2), ti=0, tf=1, parameters=(1.2, 4, 1.23), step=.00001):
    """
    A euler integrator for three co-dependent functions in the driven pendulum system,
    d/dt(omega) = -1/q*omega-sin(theta)+g*cos(phi)
    d/dt(theta) = omega
    d/dt(phi)   = omega_f
    :param ti: initial time, in seconds
    :param tf: final time, exclusive, in seconds
    :param initial_conditions: a tuple containing the initial values of omega, theta, and phi, in that order
    :param parameters: a tuple containing the parameters of the system q, g, and wd, in that order.
    q is the dampening parameter, g is the forcing amplitude, and wd is the angular velocity of the driving force.
    :param step: step size in time of the integration
    :return:
    """
    if parameters[0] == 0:
        raise Exception("the q parameter must be nonzero")
    tms = np.arange(ti, tf+step, step)
    omega = np.zeros(tms.size)
    theta = np.zeros(tms.size)
    phi = np.zeros(tms.size)
    omega[0] = float(initial_conditions[0])
    theta[0] = float(initial_conditions[1])
    phi[0] = float(initial_conditions[2])
    q = float(parameters[0])
    g = float(parameters[1])
    wd = float(parameters[2])
    for i in range(1, tms.size):
        o = omega[i-1]
        t = theta[i-1]
        p = phi[i-1]
        omega[i] = omega[i-1] + (-1./q*o-np.sin(t)+g*np.cos(p))*step
        theta[i] = theta[i-1] + o*step
        phi[i] = phi[i-1] + wd*step
    return omega, theta, phi, tms


def func(tuple, t, q, g, w):
    """
    a derivitive method used by Odeint that returns a tuple of derivatives for the driven pendulum system.
    :param tuple: the current values of (omega, theta, phi)
    :param t: time, in seconds
    :param q: the dampening parameter
    :param g: the forcing amplitude
    :param w: wd parameter, the angular velocity of the driving force
    :return:
    """
    omega = tuple[0]
    theta = tuple[1]
    phi = tuple[2]
    return -1./q*omega-np.sin(theta)+g*np.cos(phi), omega, w


def compare(initial_conditions=(0, np.pi/2, 0), ti=0, tf=1, parameters=(1.2, 4, 1.23), step=.01, projection="c"):
    """
    A method that runs both the Euler and Odeint methods of integration and compares the results by
    plotting them.
    :param ti: initial time, in seconds
    :param tf: final time, exclusive, in seconds
    :param initial_conditions: a tuple containing the initial values of omega, theta, and phi, in that order
    :param parameters: a tuple containing the parameters of the system q, g, and wd, in that order.
    q is the dampening parameter, g is the forcing amplitude, and wd is the angular velocity of the driving force.
    :param step: the step size of the integration
    :param projection: the type of plot to create with the results. can be "cartesian" or "c" for cartesian, "polar" or
    "p" for polar, "b" or "both" for both side by side, or "none" or "n" for no plot. The default is cartesian.
    :return:
    """
    e_omega, e_theta, e_phi, e_tms = pendulum_euler(ti=ti,
                                                    tf=tf,
                                                    parameters=parameters,
                                                    initial_conditions=initial_conditions,
                                                    step=step)
    resp = odeint(func, initial_conditions, e_tms, args=parameters)
    theta = np.zeros(e_tms.size)
    for i in range(e_tms.size):
        theta[i] = resp[i][1]
    if projection == "c" or projection == "cartesian":
        plot_cart((e_theta, theta), e_tms)
    if projection == "p" or projection == "polar":
        plot_polar((e_theta, theta), e_tms)
    if projection == "b" or projection == "both":
        plot_both((e_theta, theta), e_tms)


def plot_cart(y, t):
    """
    A method that plots the results of either the Euler integrator, the Odeint integrator, or both.
    :param y: the y values to plot. a tuple of (euler values, odeint values) can also be used to compare the results
    :type y: np.ndarray or tuple
    :param t: the t values to plot
    :return:
    """
    ax = plt.gca()
    ax.set_title("Oscillations of a Driven Pendulum")
    ax.set_ylabel("Theta (rad)")
    ax.set_xlabel("Time (sec)")
    if type(y) is tuple:
        plt.plot(t, y[0], c='b', label='Euler')
        plt.plot(t, y[1], c='r', label='Odeint')
    if isinstance(y, np.ndarray):
        plt.plot(t, y, c='b')
    plt.legend()
    plt.gcf().set_facecolor('w')
    plt.show()


def plot_polar(y, t):
    """
    A method that creates a polar plot of the results of either the Euler integrator, the Odeint integrator, or both.
    :param y: the y values to plot. a tuple of (euler values, odeint values) can also be used to compare the results
    :type y: np.ndarray or tuple
    :param t: the t values to plot
    :return:
    """
    ax = plt.subplot(111, projection='polar')
    plt.suptitle("Oscillations of a Driven Pendulum")
    if type(y) is tuple:
        plt.plot(y[0], t, c='b', label='Euler')
        plt.plot(y[1], t, c='r', label='Odeint')
    if isinstance(y, np.ndarray):
        plt.plot(t, y, c='b')
    plt.legend()
    plt.gcf().set_facecolor('w')
    plt.show()


def plot_both(y, t):
    """
    A method that creates both a cartesian and polar plot of the results of either the Euler integrator, the Odeint integrator, or both.
    :param y: the y values to plot. a tuple of (euler values, odeint values) can also be used to compare the results
    :type y: np.ndarray or tuple
    :param t: the t values to plot
    :return:
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(12, 10)
    if type(y) is tuple:
        axp1 = fig.add_subplot(gs[0:8, :5], projection='polar')
        axp2 = fig.add_subplot(gs[0:8, 5:], projection='polar')
        axc = fig.add_subplot(gs[9:, 1:-1])
        axc.plot(t, y[0], c='b', label='Euler')
        axc.plot(t, y[1], c='r', label='Odeint')
        axp1.plot(y[0], t, c='b', label='Euler')
        axp2.plot(y[1], t, c='r', label='Odeint')
    if isinstance(y, np.ndarray):
        axp = fig.add_subplot(gs[0:8, :], projection='polar')
        axc = fig.add_subplot(gs[9:, 1:-1])
        axc.plot(t, y, c='b')
        axp.plot(t, y, c='b')
    gs.update(wspace=0.5, hspace=0.5)
    plt.legend(loc=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (radians)')
    plt.suptitle("Oscillations of a Driven Pendulum")
    plt.gcf().set_facecolor('w')
    plt.savefig("periodic.pdf")    
    plt.show()

if __name__ == "__main__":
    compare(initial_conditions=(2, 6, 3), parameters=(2, 1, 2 / 3), tf=100, projection="b")


