import losses
import optimizers
import poly_function
from datagen import generate_data

if __name__ == "__main__":
    x, y = generate_data(3, -2, 1, 10000, 0.01)
    poly_fn = poly_function.Polynom()
    optimizer = optimizers.SGD(
        poly_fn,
        losses.MSELoss(),
        0.01,
    )
    loss_history = optimizer.fit(x, y, 1000, 64)
    print("a={:.2f}, b={:.2f}, c={:.2f}".format(*poly_fn.weights))
