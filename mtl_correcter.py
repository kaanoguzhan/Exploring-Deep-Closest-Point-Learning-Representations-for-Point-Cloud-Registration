
import glob
import fileinput
import in_place
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(*[BASE_DIR, 'data', 'mixamo', 'objfiles'])

mtl_files = glob.glob(DATA_DIR + "*.mtl")

for filename in mtl_files:
    f = open(filename, "r")
    with in_place.InPlace(filename) as file:
        for l in file:

            ed = l.rfind('\\')
            st = l.find('C:')
            name = filename[:filename.find("_")]

            if st != -1 and ed != -1:

                #l = l.replace(l[(st):ed+1],"/" + name + "_textures")
                l = l.replace(l[(st):ed+1], name + "_textures/")
                # print(l,end="")
            print(l, end="")
            file.write(l)
