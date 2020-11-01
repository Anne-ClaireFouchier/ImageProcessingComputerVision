import os
import numpy as np

def convert_line_to_size(size_line):
    size=size_line.split(" ")
    size[-1]=size[-1].split("\n")[0]
    for i in range(len(size)):
        size[i]=int(size[i])
    return size


def reduce_resolution(val_mat, intensity_max, number_values=0):
    if number_values<=0 or number_values>=intensity_max:
        return val_mat
    else:
        number_values=number_values+1 #because 0 is not a label
        border=np.floor(intensity_max/number_values)
        for i in range(number_values-1):
            val_mat[(val_mat>=border*i) & (val_mat<border*(i+1))]=border*i
        val_mat[val_mat>=border*(number_values-1)]=intensity_max
    return val_mat

def get_vertice_number(vertices_array, vertice_coord):
    # checks if the vertex exists. If so, returns the index, if not, creates it and return index.
    if vertice_coord in vertices_array:
        index=vertices_array.index(vertice_coord)
    else :
        index=len(vertices_array)
        vertices_array.append(vertice_coord)
    index=index+1 # because vertices numbers start with 1
    return index, vertices_array

def append_faces_square(v1, v2, v3, v4, faces_array):
    # appends to the faces_array the face made of 2 triangles, both sides
    faces_array.append([v1, v2, v3])
    faces_array.append([v1, v3, v2])
    faces_array.append([v3, v4, v1])
    faces_array.append([v3, v1, v4])
    return faces_array

def append_separator(vertices_array, faces_array, c1, c2, c3, c4):
    # creates/get the needed vertices, make faces, return the arrays of vertices and faces
    v1, vertices_array=get_vertice_number(vertices_array, c1)
    v2, vertices_array=get_vertice_number(vertices_array, c2)
    v3, vertices_array=get_vertice_number(vertices_array, c3)
    v4, vertices_array=get_vertice_number(vertices_array, c4)
    faces_array=append_faces_square(v1, v2, v3, v4, faces_array)
    return vertices_array, faces_array
