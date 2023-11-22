import multiprocessing


def multiprocessing_decorator(func):
    def wrapper(arguments):
        with multiprocessing.Pool() as pool:
            results = pool.map(func, arguments)
        return results
    return wrapper

if __name__ == "__main__":
    # Move the function definition inside the if __name__ block
    @multiprocessing_decorator
    def worker_function(arg):
        return arg * 2

    # Example usage
    input_data = [1, 2, 3, 4, 5]

    # Using the decorated function to apply the function in parallel
    output_result = worker_function(input_data)

    print("Input:", input_data)
    print("Output:", output_result)