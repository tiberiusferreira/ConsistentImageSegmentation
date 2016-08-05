#!/usr/bin/env python
import sys

# just replaces the dictionary of every line by the final dictionary, so there are no partial dictionaries

def preprocessfile(file):
    last_dic = ''
    lines = []
    get_next = 0
    lines_since_got = -100000
    outfile = file
    for line_num, line in enumerate(file):
        if get_next == 1:
            last_dic = line
            get_next = 0
        if '#Object' in line:
            get_next = 1
    file.seek(0)
    separated_words = last_dic.split()
    lenght_dic = len(separated_words)
    for line_num, line in enumerate(file):
        if lines_since_got == 0:
            print 'Replacing ' + line + ' with ' + last_dic
            line = line.replace(line, last_dic)
        if lines_since_got == 5:
            for x in range(len(line.split()), lenght_dic):
                line = ''.join([line.strip(), ', 0.0', '\n'])
        lines_since_got += 1
        if '#Object' in line:
            lines_since_got = 0
        lines.append(line)
    outfile.seek(0)
    for line in lines:
        outfile.write(line)



if __name__ == '__main__':
    if len(sys.argv) > 2:
        print 'Pass only the (path) to the recordings file.\n'
        exit(1)
    try:
        f = open('RecordedData/ExperimentDataLog', 'r', 0)
    except IOError:
        print ("Could not open file " + str(sys.argv[0] + "!"))
        exit(1)
    preprocessfile(f)





