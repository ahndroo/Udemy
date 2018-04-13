'''
Consider the following list:

temperatures=[10,-20,-289,100]

Then, iterate over the temperature converter function that you created in execise 3 and print out the following output.

50.0
-4.0
That temperature doesn't make sense!
212.0
'''

from codingEx3 import C2F

temps = [10, -20, -289, 100]
for t in temps:
    C2F(t)
