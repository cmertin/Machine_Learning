# This file deals with writing data to file in different formats
from __future__ import print_function, division

def Write_CSV(data, filename, precision=5):
    print("Writing to \"" + filename + "\"")
    fout = open(filename, 'w')
    temp_str = "\"%." + str(precision) + "f\""
    for d in data:
        for i in range(0, len(d)):
            temp = d[i]
            temp = temp_str % temp
            if i < len(d)-1:
                fout.write(temp + ',')
            else:
                fout.write(temp + '\n')
    fout.close()

def ReadCSV(filename, skip_first=True):
    x = []
    y = []
    lines = [line.rstrip('\n') for line in open(filename)]
    if skip_first == True:
        lines = lines[1:]

    for line in lines:
        local = [1] # First element of x = 0, offset for theta
        temp = line.split(',')
        for item in temp[:-1]:
            local.append(float(item.replace("\"","")))
        x.append(local)
        y.append(float(temp[-1].replace("\"","")))
    return x, y
