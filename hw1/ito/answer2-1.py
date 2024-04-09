if __name__ == "__main__":
    a = ["apple", 10, False]
    b = "orange"

    a += [b,]
    a[1] = 5

    print(a) # ["apple", 5, False, "orange"]
