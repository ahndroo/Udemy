# read in file with open method
file = open('example.txt','r') # file has type TextIOWrapper
file.read() # outputs lines --at end of read method, text point is at end of file
file.read() # outputs [] -- have to reset point to first line
file.seek(0) # resets pointer
content = file.readlines() # reads file again -- not sure of diff between read, readlines, readline
print(content) # '\n' new line operator displays...use:
content=[i.rstrip("\n") for i in content]
print(content)
file.close()

# writing to files
file = open('example2.txt','w') # will create file if it doesnt exist
file.write('Line 1')
file.write('Line 2') # writes 'Line 1Line 2' in file...can use \n to use new  lines
# using a for loop
l = ['Line 1', 'Line 2', 'Line 3']
file=open('example3.txt','w')
for item in l:
    file.write(l+'\n')
file.close()
# if want to use same file and just append to it use
file = open('example3.txt','a')
file.write("Line 4")
file.close()
