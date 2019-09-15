import numpy as np


def backtrack_step(f, direction, x, dom):
    alpha = 0.3
    beta = 0.3
    t = 1
    y_0 = f(x)
    y_t = f(x + direction * t)

    while not dom(x + direction * t) \
            or y_t > y_0 - alpha * np.linalg.norm(t * direction, ord=2) * np.linalg.norm(direction, ord=2):
        t = beta * t
        y_t = f(x + direction * t)
    return t, direction * t


def gradient_descent(f, gradient_f, initial_p, epsilon, dom):
    x = initial_p
    gradient_mod = np.linalg.norm(gradient_f(x), ord=2)

    iterations = 0

    while gradient_mod > epsilon:
        iterations = iterations + 1
        if iterations > 400:
            print("break due to timeout")
            break
        direction = -gradient_f(x)
        t, line_stp = backtrack_step(f, direction, x, dom)
        x = x + line_stp
        gradient_mod = np.linalg.norm(gradient_f(x), ord=2)
        print("iteraction:", iterations, gradient_mod)

    print('tolerance:', gradient_mod)
    return x, f(x)


def ch_inv(X):
    return np.linalg.inv(X)


def newton_method(f, gradient_f, hessian_f, initial_p, epsilon, dom):
    x = initial_p
    iterations = 0
    while True:
        iterations = iterations + 1
        hessian_f_inverse = ch_inv(hessian_f(x))
        jac_f = gradient_f(x)
        #print(jac_f.shape, hessian_f_inverse.shape)
        direction = -np.dot(hessian_f_inverse, jac_f)
        newton_decrement = np.dot(jac_f.T, np.dot(hessian_f_inverse, jac_f))

        if newton_decrement / 2.0 < epsilon:
            print("iteration: #", iterations)
            print("newton decrement", newton_decrement)
            print("x, t", x.T, t)
            return x, f(x)

        t, line_stp = backtrack_step(f, direction, x, dom)
        x = x + line_stp
        print("iteration: #", iterations)
        print("newton decrement", newton_decrement)
        print("x, t", x.T, t)
