# Day_05_02_PythonList.py

#             []    ()     {}   {}
# collection: list, tuple, set, dict

a = [1, 3, 7, 8, 17]
print(a)
print(a[0], a[1], a[2], a[3])
print(a[0] + a[1] + a[2] + a[3])
print()

for i in range(len(a)):
    print(i, a[i])
print()

for i in range(len(a) // 2):
    print(i, a[i])
print()

for i in a:
    print(i, end=' ')
print()

print(a[4], a[len(a) - 1], a[-1])
