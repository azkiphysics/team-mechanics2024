def fibonacci(n):
    for i in range(n):
        n += i
    return n


if __name__ == "__main__":
    answer = fibonacci(10)
    print(answer)
