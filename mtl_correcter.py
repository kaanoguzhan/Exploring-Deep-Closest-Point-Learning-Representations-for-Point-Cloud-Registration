
import glob
import fileinput
import in_place
data_path = "/home/yigittunali/Documents/Projects/ml3d/objfiles/"


mtl_files = glob.glob(data_path + "*.mtl")

#root_path =
for filename in mtl_files:
    f = open(filename,"r")
    #o =
    with in_place.InPlace(filename) as file:
        for l in file:

            ed = l.rfind('\\')
            st = l.find('C:')
            name = filename[:filename.find("_")]

            if st != -1 and ed != -1:

                #l = l.replace(l[(st):ed+1],"/" + name + "_textures")
                l = l.replace(l[(st):ed+1], name + "_textures/")
                #print(l,end="")
            print(l,end="")
            file.write(l)

