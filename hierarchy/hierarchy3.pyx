



def hierarchical_indices(length):
    indices = []
    cdef int level = length // 2
    cdef int i, j, start
    cdef float val
    while level > 0:
        for i from level <= i < length by 2 * level:
            for j from level <= j < length by 2 * level:
                indices.append((i, j))
        for i from 0 <= i < length by level:
            if i % (2 * level) == 0:
                start = level
            else:
                start = 0
            for j from start <= j < length by 2 * level:
                indices.append((i, j))
        level /= 2
    
