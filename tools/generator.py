import numpy as np
import os

SIZES = [1024, 4096, 16384, 65536, 262144]

OUT = "../data/input"

os.makedirs(OUT, exist_ok=True)


def num_blobs_for_n(n):
    if n <= 2048:
        return 4
    elif n <= 8192:
        return 6
    elif n <= 32768:
        return 8
    elif n <= 131072:
        return 10
    else:
        return 12


def write_csv(path, X, y):
    with open(path, "w") as f:
        f.write("x,y,z,label\n")
        for i in range(X.shape[0]):
            f.write(f"{X[i,0]},{X[i,1]},{X[i,2]},{y[i]}\n")

def generate_gaussian(n):
    k = num_blobs_for_n(n)
    spacing = 6.0
    sigma = 0.9

    grid = int(np.ceil(k ** (1/3)))
    centers = []
    for x in range(grid):
        for y in range(grid):
            for z in range(grid):
                centers.append([x * spacing, y * spacing, z * spacing])
                if len(centers) == k:
                    break
            if len(centers) == k:
                break
        if len(centers) == k:
            break

    centers = np.array(centers)

    pts = []
    labels = []

    per = n // k
    for i in range(k):
        cnt = per if i < k - 1 else n - per * (k - 1)
        blob = np.random.randn(cnt, 3) * sigma + centers[i]
        pts.append(blob)
        labels.append(np.full(cnt, i, dtype=int))

    return np.vstack(pts), np.concatenate(labels), k


def main():
    for n in SIZES:
        # Gaussian
        Xg, yg, k = generate_gaussian(n)
        write_csv(
            f"{OUT}/gaussian_{n}.csv",
            Xg, yg
        )

        print(f"Generated n={n}")

    print("Done.")


if __name__ == "__main__":
    main()
