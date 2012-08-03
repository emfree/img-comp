


def order2(int M):
    if M == 0:
        return 0
    cdef int i = 0
    while M % 2 == 0:
        M >>= 1
        i += 1
    return i

