import numpy as np

if __name__ == "__main__":
    vectors = [
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [-3.0, 1.0],
        [-2.0, -1.0],
        [-3.0, -2.0]
    ]
    classes = [1, 1, 1, -1, -1, -1]
    beta = [1.0, 0.5, 0.5, 0.5, 0.5, 1]
    gamma = 0.1
    
    a = np.array([1.0, 0.0, 0.0])

    vectors = np.stack(vectors)
    print(vectors.shape)
    classes = np.array(classes)
    beta = np.array(beta)

    y_samples = np.concatenate([np.ones(vectors.shape[0])[:, None], vectors], axis=1)
    y_samples = y_samples * classes[:, None]

    for i in range(2):
        for j, sample in enumerate(y_samples):
            # g(x) = a^T * yk
            g_x = np.dot(a, sample)

            # a = a - gamma * (a^Tyk - beta) * yk
            at_yk_minus_beta = g_x - beta[j]
            a = a - gamma * at_yk_minus_beta * sample

            iteration = i * len(y_samples) + j + 1
            print(f"Iteration: {iteration}, a^Tyk={g_x:.2f}, beta={beta[j]:.2f}, a={[float(f'{x:.2f}') for x in a]}")