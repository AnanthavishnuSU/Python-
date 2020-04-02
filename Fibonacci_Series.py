# FIBONACCI SERIES
# AUTHOR : ANANTHAVISHNU S U

n = 100  # Change the value of n to get n number of Fibonacci numbers
a = 0
b = 1
c = 0
print("Fibonacci Series: ")
for i in range(n):
    print(c)
    a = b
    b = c
    c = a + b
