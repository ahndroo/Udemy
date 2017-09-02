'''
In exercise 1 you created a function that converted Celsius degrees to Fahrenheit.

The lowest possible temperature that physical matter can reach is -273.15 °C. With
that in mind, please improve the function by making it print out a message in case
 a number lower than -273.15 is passed as input when calling the function.
'''

def C2F(temp_celsius):

    if temp_celsius < -273.15:
        print('Error! No physical matter in the whole entire universe is that cold!!')
    else:
        temp_Fahrenheit = temp_celsius* (9/5) + 32
        print('The temperature '+str(temp_celsius)+'C is '+str(temp_Fahrenheit)+'F')
        return temp_Fahrenheit

def main():
    F = C2F(-300)

if __name__ == '__main__':
    main()
