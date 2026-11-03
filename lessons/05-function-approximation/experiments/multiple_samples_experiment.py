"""L05: fixed-target linear fitting. Python standard library only.
No environment interaction, GPU, network, files, or unbounded loops.
The reference 2*x+1 is known only because this is a constructed example.
"""
from math import isfinite, sqrt

STEPS = 40
ALPHAS = (0.05, 0.2, 2.2)
INITIAL_WEIGHTS = (0.0, 0.0)
TRAIN_XS = (-1.0, 0.0, 1.0)
NARROW_XS = (-0.1, 0.0, 0.1)
EVAL_XS = (-0.5, 0.5, 2.0)


def reference_value(x):
    return 2.0 * x + 1.0


def make_data(xs):
    return tuple((x, reference_value(x)) for x in xs)


def predict(x, weights):
    w1, w0 = weights
    return w1 * x + w0


def loss(data, weights):
    return sum((predict(x, weights) - y) ** 2 for x, y in data) / (2 * len(data))


def rmse(data, weights):
    return sqrt(2 * loss(data, weights))


def batch_gradient(data, weights):
    # All errors use this SAME group of old weights.
    grad_w1 = 0.0
    grad_w0 = 0.0
    for x, target in data:
        error = predict(x, weights) - target
        grad_w1 += error * x
        grad_w0 += error
    n = len(data)
    return grad_w1 / n, grad_w0 / n


def step(data, weights, alpha):
    grad_w1, grad_w0 = batch_gradient(data, weights)
    w1, w0 = weights
    return w1 - alpha * grad_w1, w0 - alpha * grad_w0


def fit(data, alpha, steps=STEPS, initial=INITIAL_WEIGHTS):
    # Bounds make accidental large edits fail quickly.
    if not data or not 1 <= len(data) <= 100:
        raise ValueError("Use 1 to 100 samples.")
    if not isinstance(steps, int) or not 0 <= steps <= 200:
        raise ValueError("Use 0 to 200 updates.")
    if not isfinite(alpha) or not 0 < alpha <= 3:
        raise ValueError("Use 0 < alpha <= 3.")
    weights = tuple(initial)
    if len(weights) != 2 or not all(isfinite(v) for v in weights):
        raise ValueError("Use two finite initial parameters.")
    history = [(0, weights, loss(data, weights))]
    status = "budget_complete"
    for update in range(1, steps + 1):
        candidate = step(data, weights, alpha)
        if not all(isfinite(v) and abs(v) <= 1e6 for v in candidate):
            status = "parameter_guard_stop"
            break
        weights = candidate
        history.append((update, weights, loss(data, weights)))
    return weights, history, status


def show_run(label, data, alpha):
    weights, history, status = fit(data, alpha)
    eval_data = make_data(EVAL_XS)
    print(f"\n{label} alpha={alpha} N={len(data)} updates={len(history)-1}")
    print("update       train_loss      w1          w0")
    for update, current, current_loss in history:
        if update in (0, 1, 5, 20, STEPS):
            print(f"{update:>6} {current_loss:>16.8g} {current[0]:>11.6g} {current[1]:>11.6g}")
    used = (len(history) - 1) * len(data)
    print(f"eval_RMSE={rmse(eval_data, weights):.8g} sample_uses={used} status={status}")
    print("eval_predictions:", [round(predict(x, weights), 6) for x in EVAL_XS])
    return weights


def main():
    print("A. One-sample vs batch: start from the SAME old weights")
    pair = ((0.0, 1.0), (1.0, 3.0))
    old = (1.0, 1.0)
    one = step((pair[1],), old, 0.2)
    batch = step(pair, old, 0.2)
    for label, weights in (("old", old), ("one_sample", one), ("batch", batch)):
        errors = [round(predict(x, weights) - y, 6) for x, y in pair]
        print(label, "weights=", tuple(round(v, 6) for v in weights),
              "errors=", errors, "pair_loss=", round(loss(pair, weights), 8))

    print("\nB. Core experiment: only alpha changes")
    wide_data = make_data(TRAIN_XS)
    for alpha in ALPHAS:
        show_run("wide", wide_data, alpha)

    print("\nC. Coverage extension: only training positions change")
    print("Same alpha=.2, initialization, N=3, updates=40, evaluation positions.")
    show_run("wide", wide_data, 0.2)
    show_run("narrow", make_data(NARROW_XS), 0.2)


if __name__ == "__main__":
    main()
