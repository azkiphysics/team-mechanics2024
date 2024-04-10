if __name__ == "__main__":
    a = ("apple", 10, False)
    b = "orange"

    a = list(a)
    a.append(b)
    a[1] = 5
    a = tuple(a)

    print(a)