
#Create a pgm3d_to_obj.py program that take as arguments 
# - a pgm3d file and 
# - the number of labels, 
# and output an OBJ file containing the boundaries between thoses labels. 
import sys
import numpy as np
import tools_multiple_objects
import tools_one_object
import tools_matrix_handling


def usage():
    print("-------------------------------------------------------------------------------------")
    print("In your folder, run : python3 pgm3d_to_obj.py your_PGM3D_file.pgm3d number_of_labels")
    print("If you want to keep the original number of labels, give 0 or below")
    print("NOTE : number_of_labels doesn ot include the \"background\".")
    print("-------------------------------------------------------------------------------------")

if __name__ == '__main__':
    try :
        filename=sys.argv[1]
        nb_labels=int(sys.argv[2])
    except:
        usage()
        sys.exit(2)

    print("PROCESSING DATA...")
    tools_multiple_objects.export_PGM3D_to_OBJ_files(filename, nb_labels)
    print("DONE")
    print("Check your folder for the objects and open them with Meshlab for example! (look for files named \'label1.obj\', \'label2.obj\'...)")
