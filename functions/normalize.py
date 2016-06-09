
import math


#Normalize the vector input:vector,length of vector   output:normalized vector

def normalizeByLength(array, length):
    dist = 0
    for x in range(0,length):
        dist +=  array[x] * array[x]
    
    dist = math.sqrt(dist)
    
    for x in range(0,length):
        array[x] = array[x] / dist
    
    return array
    
    
    
def none(array,length):
    return array