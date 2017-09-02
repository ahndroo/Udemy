'''
Please create a function that converts Celsius degrees to Fahrenheit. The formula to convert Celsius to Fahrenheit is F = C × 9/5 + 32.
'''

def C2F(temp_celsius):
    temp_Fahrenheit = temp_celsius* (9/5) + 32
    print('The temperature '+str(temp_celsius)+'C is '+str(temp_Fahrenheit)+'F')
    return temp_Fahrenheit

def main():
    F = C2F(23)

if __name__ == '__main__':
    main()
