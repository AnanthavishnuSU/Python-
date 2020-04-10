import secrets as sc

Char_List = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
             "!", "@", "#", "$", "%", "^", "&", "*", "(", ")",
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
             't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def password_gen(k):
    a = []
    for i in range(k):
        a.append(sc.choice(Char_List))
    b = ""
    print(b.join(a))


n = int(input("Enter the number of characters required: "))
w = int(input("Enter number of different passwords required: "))

print("The Passwords:")
for h in range(w):
    password_gen(n)
