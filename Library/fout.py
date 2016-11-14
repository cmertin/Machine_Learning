# This file deals with writing data to file in different formats
from __future__ import print_function, division

def Write_CSV(data, filename, precision=5):
    print("Writing to \"" + filename + "\"")
    fout = open(filename, 'w')
    temp_str = "\"%." + str(precision) + "f\""
    for d in data:
        for i in range(0, len(data)):
            temp = d[i]
            temp = temp_str % temp
            if i < len(data)-1:
                fout.write(temp + ',')
            else:
                fout.write(temp + '\n')
    fout.close()
