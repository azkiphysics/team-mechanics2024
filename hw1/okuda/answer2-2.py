a = ("apple", 10, False)
b = "orange"

a += (b,)
a = list(a)
a[1] = 5

print(a)
