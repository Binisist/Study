##소인수분해 - 내장함수 없이 구현!!

##주어진 수 약수 후보 -> prime!!
import math
user = int(input())
b = int(math.sqrt(user))+1
prime = []

for num in range(2,b+1):
    check = 0
    for i in range(2,int(math.sqrt(num))+1):
        if num % i == 0:
            check += i

    if check == 0:
        prime += [num]

##각 소수별 지수 찾기 -> exp
exp = [0 for i in range(len(prime))]

while user > 1:
    for i in range(len(prime)):
        if user % prime[i] == 0:
            exp[i] += 1
            user = user / prime[i]
            continue
        else:
            pass

print(prime,exp)