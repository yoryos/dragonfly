from Vectors import Vector_2D
import numpy as np

class Pixel_Array(object):

    a = None

    def __init__(self, array, middle_width, middle_height):

        self.a = array
        self.middle_width = middle_width
        self.middle_height = middle_height
        self.height, self.width = self.a.shape

    def get(self, r,c):

        return self.a[r][c]

    def get_range(self, r_min, r_max, c_min, c_max):

        return [[self.a[r][c] for r in range(r_min,r_max)] for c in range(c_min,c_max)]

    def get_top(self):

        left_array = []
        right_array = []
        for i in xrange(self.height/2):
            if i < (self.height - self.middle_height) / 2:
                width = self.width / 2
            else:
                width = (self.width - self.middle_width) / 2
            left_row = self.a[i][:width]
            right_row = self.a[i][-width:]
            left_array.append(left_row)
            right_array.append(right_row)
        return left_array, right_array

    def get_bottom(self):

        left_array = []
        right_array = []
        for i in xrange(self.height/2, self.height):
            if i < (self.height / 2 + self.middle_height/2):
                width = (self.width - self.middle_width) / 2
            else:
                width = self.width / 2
            left_row = self.a[i][:width]
            right_row = self.a[i][-width:]
            left_array.append(left_row)
            right_array.append(right_row)

        return left_array, right_array

    def get_quadrants(self):

        return self.get_top() + self.get_bottom() + self.get_middle()
    def get_middle(self):

        t = self.height/2 - self.middle_height/2
        b = self.height/2 + self.middle_height/2
        l = self.width/2 - self.middle_width/2
        r = self.width/2 + self.middle_width/2

        return self.a[t:b,l:r],

    def __str__(self):
        return str(self.a)

    @staticmethod
    def flatten(array):
        b = []
        for i in array:
            for j in i:
                b.append(j)
        return b

#
# ar = Pixel_Array(np.array([[Vector_2D(i,j) for j in xrange(12)] for i in xrange(6)]), 8,3)
#
# tl,tr = ar.get_top()
# bl,br = ar.get_bottom()
#
# a,b,c,d,e = ar.get_quadrants()
#
#
# print a
# print Pixel_Array.flatten(a)
#
# # print a
# print b
# print c
# print d
# print e
