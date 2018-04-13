'''
Here's a rather challenging exercise that integrates functions, loops, conditionals, and file handling.

In exercise 4, you recursively printed out converted temperature in the command line. For this exercise, please consider the same list of Celsius values again as input:

temperatures=[10,-20,-289,100]

Try to make a script that converts Celsius to Fahrenheit and creates a text file and stores the converted values inside the text file. Your text file content should look like this:

50.0
-4.0
212.0

Please don't write any message in the text file when input is lower than -273.15.

'''

def C2F(temp_celsius):
    if temp_celsius > -273.15:
        temp_Fahrenheit = temp_celsius* (9/5) + 32
        # print('The temperature '+str(temp_celsius)+'C is '+str(temp_Fahrenheit)+'F')
        return temp_Fahrenheit
    else:
        return

def main():

    temperatures=[10,-20,-289,100]

    fname = input("Please enter filename: ")
    with open(fname + '.txt','w') as file:
        for t in temperatures:
            temp = C2F(t)
            if temp != None:
                file.write(str(C2F(t)) + '\n')

if __name__ == '__main__':
    main()
