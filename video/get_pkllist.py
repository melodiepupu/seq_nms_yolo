import os
import shutil

count = 0
for element in os.listdir('input'):
	if(str(element).endswith('jpg')):
	    count = count + 1

string = ''
for i in range(count):
    string += 'video/input/frame{}.jpg\n'.format(i)

with open('pkllist.txt', 'w') as f:
    f.write(string)
