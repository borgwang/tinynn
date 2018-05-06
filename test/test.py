array = [1,2,3,4,5]

for a in array:
    print(a, id(a))
    a -= 1
    print(a, array)
    print('---')
