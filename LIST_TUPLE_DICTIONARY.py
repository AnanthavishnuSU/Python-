# LIST

L1 = ["Amit", "Sanjay", "Nikhil"]
L2 = ["Intel", "AMD", "Samsung"]
print("LIST:")
print("L1: ", L1)
print("L2: ", L2)
a = input("Enter name:")
k = input("Enter the name of list(L1 or L2): ")
if k == "L1":
    L1.append(a)
else:
    L2.append(a)
print("CURRENT LIST:")
print("L1: ", L1)
print("L2: ", L2)

# Tuple

T1 = ("SLV", "ASLV", "PSLV", "GSLV")
b = input("Enter the rocket to be searched:")
m = 0
for i in T1:
    if b == i:
        m = m + 1
        break
    else:
        m = 0
if m == 0:
    print("NOT FOUND")
else:
    print(b + " is an element of tuple")

# Dictionary

D1 = {
    "SLV": "SATELLITE LAUNCH VEHICLE",
    "ASLV": "AUGMENTED SATELLITE LAUNCH VEHICLE",
    "PSLV": "POLAR SATELLITE LAUNCH VEHICLE",
    "GSLV": "GEOSYNCHRONOUS SATELLITE LAUNCH VEHICLE"
}
print("Current dictionary:")
print(D1)
c = input("Enter the element to be removed:")
D1.pop(c)
print("Current dictionary:")
print(D1)
