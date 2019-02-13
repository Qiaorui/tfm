def add(x, y):
    return x + y


def minos(x, y):
    if x <


    z = x if x < y else 0



    while z + y != x:
        z += 1
    return z


def mul(x, y):
    z = 0
    pos = True

    if x < 0 and y < 0:
        x = minos(0, x)
        y = minos(0, y)
    elif y < 0:
        y = minos(0, y)
        pos = False

    for _ in range(y):
        z += x

    if pos:
        z = minos(0, z)

    return z


def div(x, y):
    if y == 0:
        return 'Not-defined'

    pos = True

    if x < 0 and y < 0:
        x = minos(0, x)
        y = minos(0, y)
    elif y < 0:
        y = minos(0, y)
        pos = False
    elif x < 0:
        x = minos(0, x)
        pos = False

    z = 0
    while mul(y, z) < x:
        z += 1

    z = z if pos else minos(0, z)

    if mul(y, z) != x:
        return 'Non-integral answer'

    return z


def power(x, y):
    if y < 0:
        return 'Non-integral answer'
    z = 1
    for _ in range(y):
        z = mul(z, x)
    return z


def compute_operation(x, op, y):
    res = ''
    if op == '+':
        res = add(x, y)
    if op == '-':
        res = minos(x, y)
    if op == '^':
        res = power(x, y)
    if op == '*':
        res = mul(x, y)
    if op == '/':
        res = div(x, y)

    print(res)
