import datetime
import os

fnames = os.listdir('sampleFiles/')
fnames.sort()
p = 'sampleFiles/'

t = datetime.datetime.now()
newFile = open(t.strftime('%Y-%m-%d-%H-%M-%s-%f'),'w')

for f in fnames:
    file = open(p + f, 'r')
    data = file.readlines()
    for i in range(len(data)):
        newFile.write(data[i] + '\n')
    file.close()

newFile.close()
