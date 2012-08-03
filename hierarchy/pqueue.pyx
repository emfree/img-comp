import numpy as np
cimport numpy as np


# priority queue


        
cdef class HeapItem:
    cdef public bint flag
    cdef public double priority
    def __init__(HeapItem self, double priority, bint flag):
        self.priority = priority
        self.flag = flag
   

cdef class Heap:

    cdef public Py_ssize_t n, space
    cdef list heap  

    def __cinit__(Heap self, int init_space=256):
        self.heap = [0]*init_space
        self.space = init_space
        self.n = 0
        
        
    cdef object push(Heap self, HeapItem item):

        cdef Py_ssize_t i
        cdef HeapItem t
       
        i = self.n
        self.n += 1

        # We need to do this because the Python C API
        # has no fast version of list.pop, only list.append,
        # otherwise we could just have appended
        if self.space >= i:
            self.heap[i] = item
        else:
            self.heap.append(item)
            self.space += 1        
            
        while i>0 and self.heap[i].priority < self.heap[(i-1)//2].priority:
            t = self.heap[(i-1)//2]
            self.heap[(i-1)//2] = self.heap[i]
            self.heap[i] = t
            i = (i-1)//2
        
        return None # for propagating exceptions
        

    cdef HeapItem peek(Heap self):
        return self.heap[0]
       

    cdef HeapItem pop(Heap self):
        
        cdef HeapItem it, t
        cdef Py_ssize_t i, j, k, l
        
        it = self.heap[0]
        
        self.heap[0] = self.heap[self.n-1]
        self.n -= 1
        
        # Since Python C API has no way of popping off a list,
        # we must fake this with slicing.
        if self.n < self.space//4 and self.space>40: #FIXME: magic number
            self.heap = self.heap[ :self.space//2+1 ]
            self.space = self.space//2+1
        
        i=0
        j=1
        k=2
        while ((j<self.n and 
                    self.heap[i].priority > self.heap[j].priority or
                k<self.n and 
                    self.heap[i].priority > self.heap[k].priority)):
            if k<self.n and self.heap[j].priority>self.heap[k].priority:
                l = k
            else:
                l = j
            t = self.heap[l]
            self.heap[l] = self.heap[i]
            self.heap[i] = t
            i = l
            j = 2*i+1
            k = 2*i+2

        return it


def codelength(np.ndarray[np.float_t, ndim = 1] dist, index):
    cdef Heap heap = Heap()
    cdef int i = 0
    cdef int count = 0
    cdef HeapItem left
    cdef HeapItem right
    cdef HeapItem new
    for i in range(256):
        heap.push(HeapItem(dist[i], i == index))
    while heap.n > 1:
        left, right = heap.pop(), heap.pop()
        new = HeapItem(left.priority + right.priority, left.flag or right.flag)
        if new.flag:
            count += 1
        heap.push(new)
    return count


