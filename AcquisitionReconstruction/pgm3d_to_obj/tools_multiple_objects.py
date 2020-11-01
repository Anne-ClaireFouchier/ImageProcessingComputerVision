import os
import sys
import numpy as np
import random
import tools_matrix_handling


def get_separating_vertices_and_faces_in_dict(val_mat):
    # goes over each pixel and its 3 forward neighbors. 
    # If 2 neighbors have different intensities, then a separator is created for both objects (except background)
    # returns the dictionaries of vertices and faces, which make the separators
    
    # get the sorted intensity range
    range_intensities=np.sort(np.unique(val_mat))
    
    # if the first visited voxel is not background, then the matrix is padded with background
    if val_mat[0, 0, 0]!=range_intensities[0]:
        m,n,p=val_mat.shape
        mat=np.full((m+2, n+2, p+2), range_intensities[0])
        mat[1:-1, 1:-1, 1:-1]=val_mat

    # create dictionaries where the vertices and faces will be stored for each object    
    size=val_mat.shape
    dict_vertices_array={}
    dict_faces_array={}
    
    #initialize each array (MUST CHECK NOT EMPTY WHEN WE WRITE FILE!!)
    for i in range(1, len(range_intensities), 1): # first label is background
        dict_vertices_array[i]=[]
        dict_faces_array[i]=[]
       
    
    # for (x, y, z), check (x+1, y, z), (x, y+1, z) and (x, y, z+1)
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                
                label=val_mat[x, y, z]
                index_curr_label=np.where(range_intensities==label)[0][0]
                         
                if x<size[0]-1:
                    if label != val_mat[x+1, y, z]:
                        index_neigh_label=np.where(range_intensities==val_mat[x+1, y, z])[0][0]
                        # vertices of the separation square --> get their vertice index
                        c1=[x+0.5, y-0.5, z-0.5]
                        c2=[x+0.5, y-0.5, z+0.5]
                        c3=[x+0.5, y+0.5, z+0.5]
                        c4=[x+0.5, y+0.5, z-0.5]
                        # face will be (v1, v2, v3) and (v3, v4, v1)
                        if index_curr_label !=0:
                            dict_vertices_array[index_curr_label], dict_faces_array[index_curr_label]=tools_matrix_handling.append_separator(dict_vertices_array[index_curr_label], dict_faces_array[index_curr_label], c1, c2, c3, c4)
                        if index_neigh_label !=0:
                            dict_vertices_array[index_neigh_label], dict_faces_array[index_neigh_label]=tools_matrix_handling.append_separator(dict_vertices_array[index_neigh_label], dict_faces_array[index_neigh_label], c1, c2, c3, c4)   

                if y<size[1]-1:
                    if label != val_mat[x, y+1, z]:
                        index_neigh_label=np.where(range_intensities==val_mat[x, y+1, z])[0][0]
                        c1=[x-0.5, y+0.5, z-0.5] 
                        c2=[x+0.5, y+0.5, z-0.5]
                        c3=[x+0.5, y+0.5, z+0.5]
                        c4=[x-0.5, y+0.5, z+0.5]
                        if index_curr_label !=0:
                            dict_vertices_array[index_curr_label], dict_faces_array[index_curr_label]=tools_matrix_handling.append_separator(dict_vertices_array[index_curr_label], dict_faces_array[index_curr_label], c1, c2, c3, c4)
                        if index_neigh_label !=0:
                            dict_vertices_array[index_neigh_label], dict_faces_array[index_neigh_label]=tools_matrix_handling.append_separator(dict_vertices_array[index_neigh_label], dict_faces_array[index_neigh_label], c1, c2, c3, c4)   

                if z<size[2]-1:
                    if label != val_mat[x, y, z+1]:
                        index_neigh_label=np.where(range_intensities==val_mat[x, y, z+1])[0][0]
                        c1=[x-0.5, y-0.5, z+0.5]
                        c2=[x+0.5, y-0.5, z+0.5]
                        c3=[x+0.5, y+0.5, z+0.5]
                        c4=[x-0.5, y+0.5, z+0.5]
                        if index_curr_label !=0:
                            dict_vertices_array[index_curr_label], dict_faces_array[index_curr_label]=tools_matrix_handling.append_separator(dict_vertices_array[index_curr_label], dict_faces_array[index_curr_label], c1, c2, c3, c4)
                        if index_neigh_label !=0:
                            dict_vertices_array[index_neigh_label], dict_faces_array[index_neigh_label]=tools_matrix_handling.append_separator(dict_vertices_array[index_neigh_label], dict_faces_array[index_neigh_label], c1, c2, c3, c4)   
                
    return dict_vertices_array, dict_faces_array




def data_to_files(dict_vertices_array, dict_faces_array):
    
    #mtl file
    mtl_filename="textures.mtl"
    m = open(mtl_filename, "w")
    
    
    #create dictionnary of colors
    seed_nb=0
    object_number=1 # in case some intensity levels are missing (so that the name of object is incrementing correctly)
    
    for new_obj in range(1, len(dict_faces_array)+1, 1):
        
        vertice_arr=dict_vertices_array[new_obj]
        face_arr=dict_faces_array[new_obj]
        
        if type(vertice_arr) is not np.ndarray:
            vertice_arr=np.array(vertice_arr)
        if type(face_arr) is not np.ndarray:
            face_arr=np.array(face_arr)
        
        # if there is not object at that intensity
        if vertice_arr.size==0:
            continue

        try:
            assert(vertice_arr.shape[1]==3)
        except :
            print("Expected : 3 coordinates per vertex")
            sys.exit(1)

        try:
            assert face_arr.shape[1]==3 
        except :
            print("Expected : 3 vertices per face")
            sys.exit(1)
        
        # create a new object file
        f = open("label"+str(object_number)+".obj", "w")
        f.write("mtllib "+mtl_filename+"\n")
        
        
        #create_color
        color=[]
        for c in range(3):
            random.seed(seed_nb)
            color.append(random.randint(0.0, 1000.0)/1000.0)
            seed_nb=seed_nb+1

        #add it to the mtl file
        mat_name="mat"+str(object_number)
        m.write("newmtl "+mat_name+"\n")
        m.write("    Kd "+str(color[0])+" "+str(color[1])+" "+str(color[2])+"\n")
                
        # write vertices and faces
        
        for i in range(vertice_arr.shape[0]):
            f.write("v "+str(vertice_arr[i, 0])+" "+str(vertice_arr[i, 1])+" "+str(vertice_arr[i, 2])+"\n")
            
        # add the mtl texture  
        f.write("usemtl "+mat_name+"\n")
        
        for j in range(face_arr.shape[0]):
            f.write("f "+str(face_arr[j, 0])+" "+str(face_arr[j, 1])+" "+str(face_arr[j, 2])+"\n")
        
        f.close()
        object_number=object_number+1
            
    m.close()



def export_PGM3D_to_OBJ_files(filename_in, number_labels=0):
    # takes a PGM3D, checks right type, extract the matrix, processes it, saves the object in an .obj file
    # returns the arrays of vertices and faces
    
    # check right format
    try :
        assert(filename_in.endswith('.pgm3d'))
    except :
        print("File should have extension .pmg3")
        sys.exit(1)

    
    data=open(filename_in)
    
    # check right header
    file_format=data.readline()

    try :
        assert(file_format=='PGM3D\n')
    except :
        print("This file has an incorrect file_format")
    
    # check size size
    size_line=data.readline()
    size=tools_matrix_handling.convert_line_to_size(size_line)

    try :
        assert(len(size)==3) 
    except :
        print("This file doesn't contain a 3D object")
    
    # get max intensity
    intensity_max=int(data.readline())
    
    # get values
    val=np.loadtxt(filename_in, dtype=np.int32, skiprows=3)
    val_mat=val.reshape([size[0], size[1], size[2]])
    
    # update resolution
    val_mat=tools_matrix_handling.reduce_resolution(val_mat, intensity_max, number_labels)
    
    # Process the values into an array of vertices and an array of faces
    vertices_array, faces_array=get_separating_vertices_and_faces_in_dict(val_mat)
    
    # make the .obj files
    data_to_files(vertices_array, faces_array)
    return vertices_array, faces_array

