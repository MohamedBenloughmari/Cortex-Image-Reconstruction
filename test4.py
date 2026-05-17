from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

if __name__ == "__main__":

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(square, range(10)))

    print(results)