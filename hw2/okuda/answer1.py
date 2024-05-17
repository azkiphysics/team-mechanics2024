while True:
    x = input("Please enter an integer: ")
    try:
        assert "." not in x
        x = int(x)
        break
    except (ValueError, AssertionError):
        print(f"{x} is not an integer.")
        continue


if x < 0:
    print(0)
elif x < 1:
    print(1)
else:
    print(2)
