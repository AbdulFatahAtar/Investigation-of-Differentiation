import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======== Part 3 – Numerical Computation ========
# This function calculates the numerical derivative of a given function using the finite difference method.
def differentiate(func, num_points, start, end):
    # Compute the step size h for evenly spacing num_points points between start and end
    h = (end - start) / (num_points - 1)
    xs = [start + i*h for i in range(num_points)]
    ys = [func(x) for x in xs]
    dys = []
    for i in range(num_points):
        if i == num_points - 1:
            d = (ys[i] - ys[i-1]) / h
        else:
            d = (ys[i+1] - ys[i]) / h
        dys.append(d)
    return xs, ys, dys

# this 5 functions are the analytical derivatives of the functions in tasks2
def d_sin5(x):
    return 5 * math.cos(5 * x)
def d_log(x):
    return 1 / x
def d_exp(x):
    return -2 * math.exp(x)
def d_poly(x):
    return 2 * x - 2
def d_composite(x):
    u = math.sqrt(x**4 + 1)
    v = math.exp(math.cos(x**2))
    du = (2 * x**3) / u
    dv = -2 * x * math.sin(x**2) * v
    return du * v + u * dv

# this is the list of functions to be differentiated, the list has their analytical derivatives, and the range of x values
tasks2 = [
    (lambda x: math.sin(5*x),              d_sin5,      1000, -math.pi, math.pi,     "sin(5x)"),
    (lambda x: math.log(x/2),              d_log,       1000, 0.5,      5,            "ln(x/2)"),
    (lambda x: -2*math.exp(x),             d_exp,       1000, -4,       0,            "-2 e^x"),
    (lambda x: x**2 - 2*x + 16,            d_poly,      1000, -5,       5,            "x^2 - 2x + 16"),
    (lambda x: math.sqrt(x**4+1)*math.exp(math.cos(x**2)), d_composite,1000, -5, 5, "√x^4+1 · e^cos(x^2)"),
]

# for loop through the functions and calculate their numerical derivatives
for func, dfunc, n, a, b, name in tasks2:
    xs, ys, dys = differentiate(func, n, a, b)
    df = pd.DataFrame({
        'x': xs,
        'f(x)': ys,
        "f'(x)_num": dys,
        "f'(x)_ana": [dfunc(x) for x in xs],
    })
    # this calculate the absolute error
    df['abs_error'] = (df["f'(x)_num"] - df["f'(x)_ana"]).abs()

    # this print the first and last 5 points of the dataframe
    print(f"=== Function: {name} ===")
    print("First 5 points:")
    print(df.head(), "\n")
    print("Last 5 points:")
    print(df.tail(), "\n")
    print(f"Mean absolute error = {df['abs_error'].mean():.4e}\n\n")


# ======== Part 4 – Graphical Analysis ========
# This function plots the original function and its derivative
orig_funcs = [
    ("sin(5x)",             lambda x: math.sin(5*x),                        -math.pi, math.pi),
    ("ln(x/2)",             lambda x: math.log(x/2),                         0.5,      5),
    ("-2e^x",               lambda x: -2*math.exp(x),                        -4,       0),
    ("x^2 - 2x + 16",       lambda x: x**2 - 2*x + 16,                       -5,       5),
    ("√x^4+1 · e^cos(x^2)", lambda x: math.sqrt(x**4+1)*math.exp(math.cos(x**2)), -5, 5),
]
# this is the 5 derivative of the original 5 functions
deriv_funcs = [
    ("5cos(5x)",            lambda x: 5*math.cos(5*x),                        -math.pi, math.pi),
    ("1/x",                 lambda x: 1/x,                                    0.5,      5),
    ("-2e^x",               lambda x: -2*math.exp(x),                        -4,       0),
    ("2x - 2",              lambda x: 2*x - 2,                               -5,       5),
    ("2x³/√x^4+1·e^cos(x^2) + √x^4+1·-2x sin(x^2)·e^cos(x^2)", 
        lambda x: (2*x**3/math.sqrt(x**4+1))*math.exp(math.cos(x**2)) 
                  + math.sqrt(x**4+1)*(-2*x*math.sin(x**2))*math.exp(math.cos(x**2)),
        -5, 5),
]
# this loop goes through the original and derivative functions and plots them and use for loop to creates 1000 points between a and b and calculates the function values for each point
n_points = 1000
for (label_o, f_o, a, b), (label_d, f_d, _, _) in zip(orig_funcs, deriv_funcs):
    xs = [a + i*(b - a)/(n_points - 1) for i in range(n_points)]
    ys = [f_o(x) for x in xs]
    plt.figure()
    plt.plot(xs, ys, color='blue')
    plt.title(f"Original: {label_o}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    
    yd = [f_d(x) for x in xs]
    plt.figure()
    plt.plot(xs, yd, color='green')
    plt.title(f"Derivative: {label_d}")
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.grid(True)
# this show the 10 plots
plt.show()


# ======== Part 5 – Vectorization ========
# This function computes the numerical derivative of a function using vectorization
def vector_diff(func, num_points, start, end):
    # Compute the step size h for evenly spacing num_points points between start and end
    h = (end - start) / (num_points - 1)
    x_arr = np.linspace(start, end, num_points)
    y_arr = func(x_arr)
    y_next = func(x_arr + h)
    d_arr = (y_next - y_arr) / h
    return x_arr, y_arr, d_arr

# prepare the list of (loop_func, vector_func, name, start, end)
timing_tasks = [
    (lambda x: math.sin(5*x),              lambda x: np.sin(5*x),                        "sin(5x)",             -math.pi, math.pi),
    (lambda x: math.log(x/2),              lambda x: np.log(x/2),                        "ln(x/2)",             0.5,      5),
    (lambda x: -2*math.exp(x),             lambda x: -2*np.exp(x),                       "-2 e^x",             -4,       0),
    (lambda x: x**2 - 2*x + 16,            lambda x: x**2 - 2*x + 16,                    "x^2 - 2x + 16",       -5,       5),
    (lambda x: math.sqrt(x**4+1)*math.exp(math.cos(x**2)),
     lambda x: np.sqrt(x**4+1)*np.exp(np.cos(x**2)),
                                             "√(x^4+1)·e^(cos(x^2))", -5, 5),
]

results = []
for loop_func, vec_func, name, a, b in timing_tasks:
    for n in [1_000, 10_000, 100_000]:
        t0 = time.time()
        differentiate(loop_func, n, a, b)
        loop_time = time.time() - t0

        t1 = time.time()
        vector_diff(vec_func, n, a, b)
        vec_time = time.time() - t1

        results.append((name, n, loop_time, vec_time, loop_time/vec_time))

# assemble into DataFrame and display
df_perf = pd.DataFrame(results, columns=['function', 'n', 'loop_time_s', 'vector_time_s', 'speedup'])
print("=== Performance Comparison for All Functions ===")
print(df_perf)




