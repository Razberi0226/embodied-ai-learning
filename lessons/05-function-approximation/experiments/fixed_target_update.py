"""一个固定target的一次gradient step；无完整RL训练、无长循环。"""


def value_estimate(x, weights):
    w1, w0 = weights
    return w1 * x + w0


def squared_loss(prediction, target):
    return 0.5 * (prediction - target) ** 2


def gradient_step(x, target, weights, alpha=0.1):
    w1, w0 = weights
    prediction = value_estimate(x, weights)
    error = prediction - target
    grad_w1 = error * x
    grad_w0 = error
    # 两项gradient均使用旧parameters；target在这一步固定。
    new_weights = (w1 - alpha * grad_w1, w0 - alpha * grad_w0)
    trace = {"prediction": prediction, "error": error,
             "gradient": (grad_w1, grad_w0),
             "loss_before": squared_loss(prediction, target),
             "loss_after": squared_loss(value_estimate(x, new_weights), target)}
    return new_weights, trace


if __name__ == "__main__":
    x, target, alpha = 1.0, 5.0, 0.1
    old_weights = (2.0, 1.0)
    new_weights, trace = gradient_step(x, target, old_weights, alpha)
    print("old weights:", old_weights)
    print("old prediction:", trace["prediction"])
    print("error:", trace["error"])
    print("gradient:", trace["gradient"])
    print("new weights:", new_weights)
    print(f"new prediction: {value_estimate(x, new_weights):.6f}")
    print(f"loss: {trace['loss_before']:.6f} -> {trace['loss_after']:.6f}")
    print("不同states的estimates也随共享parameters变化：")
    for query in (0, 1, 2):
        print(query, f"{value_estimate(query, old_weights):.6f}",
              "->", f"{value_estimate(query, new_weights):.6f}")
    print("target是给定训练目标；本演示未提供所有states的真实value。")
