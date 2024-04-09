if __name__ == "__main__":
    a = ("apple", 10, False)
    b = "orange"

    a += (b,)
    a = list(a)
    a[1] = 5
    a = tuple(a)

    print(a) # ("apple", 5, False, "orange")
