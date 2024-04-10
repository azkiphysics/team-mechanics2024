if __name__ == "__main__":
    # 整数の入力
    while True:
        x = input("Please enter an integer: ")
        if isinstance(x, int):
            break

    if x < 0:
        print(0)
    elif x < 1:
        print(1)
    else:
        print(2)
