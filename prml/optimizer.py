import numpy as np


def backtrack_step(f, gradient_f, direction, x, dom):
    alpha = 0.3
    beta = 0.3
    t = 1.0
    y_0 = f(x)
    y_t = f(x + direction * t)

    while not dom(x + direction * t) \
            or y_t > y_0 + alpha * np.linalg.norm(t * gradient_f(x), ord=2) * np.linalg.norm(direction, ord=2):
        t = beta * t
        y_t = f(x + direction * t)
    return t, direction * t


def gradient_descent(f, gradient_f, initial_p, epsilon, dom, max_iterations=200):
    x = initial_p
    gradient_mod = np.linalg.norm(gradient_f(x), ord=2)

    iterations = 0

    while gradient_mod > epsilon:
        if iterations > max_iterations:
            print(f"max iteration limit ({max_iterations}) gets hit")
            break
        direction = -gradient_f(x)
        t, line_stp = backtrack_step(f, gradient_f, direction, x, dom)
        x = x + line_stp
        gradient_mod = np.linalg.norm(gradient_f(x), ord=2)
        iterations = iterations + 1
        print(f"iteration #{iterations}: gradient_mod = {gradient_mod}, x = {x}")

    return x, f(x)


def ch_inv(X):
    return np.linalg.inv(X)


def newton_method(f, gradient_f, hessian_f, initial_p, epsilon, dom, max_iterations=200):
    x = initial_p
    iterations = 0
    while True:
        if iterations > 400:
            return x, f(x)
        iterations = iterations + 1
        hessian_f_inverse = ch_inv(hessian_f(x))
        jac_f = gradient_f(x)
        direction = -np.dot(hessian_f_inverse, jac_f)
        newton_decrement = np.dot(jac_f.T, np.dot(hessian_f_inverse, jac_f))
        print(f"newton_direction (iteration {iterations}): {direction} with newton"
              f" decrement {newton_decrement}")

        if newton_decrement / 2.0 < epsilon:
            return x, f(x)

        t, line_stp = backtrack_step(f, gradient_f, direction, x, dom)
        x = x + line_stp
        print(f"x({x.T}), f(x){f(x)}, gradient_f(x)({gradient_f(x)}),"
              f" hessian_f(x)({hessian_f(x)})")
        print(f"iteration #{iterations}, line step = {line_stp},"
              f" step length = {t}, x={x}, newton decrement = {newton_decrement}")
        if iterations % 100 == 0:
            print(f"iteration #{iterations}, x = {x.T}, t = {t}, f(x) = {f(x)},"
                  f" gradient_f(x) = {gradient_f(x)}, hessian_f(x) = {hessian_f(x)}")


def ip_f_dector(f, t, A, b):

    def new_f(x):
        return f(x) * t + np.sum(-np.log(-np.dot(A, x) + b))

    return new_f


def ip_grad_f_dector(grad_f, t, A, b):

    def new_grad_f(x):
        return t * grad_f(x) + np.sum([A_row / (-np.dot(A_row, x) + b_row) for A_row, b_row in zip(A, b)], axis=0)

    return new_grad_f


def ip_hessian_f_dector(hessian_f, t, A, b):
    def new_hessian_f(x):
        res = t * hessian_f(x)
        for A_i in range(A.shape[0]):
            tmp = np.zeros((A.shape[1], A.shape[1]))
            for A_jj in range(A.shape[1]):
                for A_kk in range(A.shape[1]):
                    tmp[A_jj, A_kk] = A[A_i, A_jj] * A[A_i, A_kk] / (-np.dot(A[A_i], x) + b[A_i]) ** 2
            res += tmp
        return res
    return new_hessian_f



# with only linear constraints
def ip_optimization(obj_f, obj_grad_f, obj_hessian_f, initial_p, A, b, epsilon=1.0e-6):
    """
    with constraints Ax - b < 0
    -1/t * (ln(-f_1(x)) + ln(-f_2(x)) + ...)
    :return:
    """

    def dom_tmp(x):
        return (np.dot(A, x) < b).all()

    t = 1.0

    m = len(A)
    print(">>>>t, obj_f, grad_f, hessian_f", t, obj_f(initial_p), obj_grad_f(initial_p), obj_hessian_f(initial_p))
    while True:
        b_obj_f = ip_f_dector(obj_f, t, A, b)
        b_obj_grad_f = ip_grad_f_dector(obj_grad_f, t, A, b)
        b_obj_hessian_f = ip_hessian_f_dector(obj_hessian_f, t, A, b)

        print(f"t({t}), x({initial_p}), obj_f({b_obj_f(initial_p)}), grad_f({b_obj_grad_f(initial_p)}), hessian_f({b_obj_hessian_f(initial_p)})")
        print("m/t", m / t)

        initial_p, _ = newton_method(b_obj_f, b_obj_grad_f, b_obj_hessian_f, initial_p, dom=dom_tmp, epsilon=epsilon)

        if m / t < 1.0e-6:
            return initial_p
        else:
            t *= 20.



