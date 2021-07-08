import sys
import random

n = int(sys.argv[1])
trials = int(sys.argv[2])
deadEnds = 0

for t in range(trials):
    m = [[False] * n for i in range(n)]
    i, j = n // 2, n // 2
    m[i][j] = True
    while 0 < i < n - 1 and 0 < j < n - 1:
        if m[i][j - 1] and m[i + 1][j] and m[i][j + 1] and m[i - 1][j]:
            deadEnds += 1
            break
        rand = random.randrange(4)
        if rand == 0 and not m[i][j - 1]:
            m[i][j - 1] = True
            j -= 1
        elif rand == 1 and m[i + 1][j] is False:
            m[i + 1][j] = True
            i += 1
        elif rand == 2 and m[i][j + 1] is False:
            m[i][j + 1] = True
            j += 1
        elif rand == 3 and m[i - 1][j] is False:
            m[i - 1][j] = True
            i -= 1

print("Dead End: {}%".format(deadEnds*100//trials))
