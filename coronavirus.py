
import json
import re
import fileinput
from matplotlib import pyplot as plt
c = 0 
dic_regioni = {}
for line in fileinput.input([r"C:\Users\massi\OneDrive\Desktop\coronavirus.txt"]):
#    print(c)
    line = line.lstrip()
    line = line.rstrip()
    line = re.sub(" +"," ",line)

    lista = line.split(",")
    for kkk in lista:
        num1 = [int(s) for s in kkk.split() if s.isdigit()]
        num = num1
        # print(num)
        # print(kkk)
        kkk2 = re.sub(str(num), "", kkk)
        kkk2 = re.sub(" +", "", kkk2)
        kkk2 = kkk2.lower()
        if kkk2 not in dic_regioni.keys():
            dic_regioni[kkk2] = {}
        dic_regioni[kkk2][c] = num[0]
    c += 1

dic_2 = {}
plt.figure()
leg = []

for key_reg in dic_regioni.keys():
    kk = dic_regioni[key_reg].keys()
    # print(list(kk))
    init = int(list(kk)[0])
    cc = []
    cc2 = []
    for a in range(init):
        cc.append(0)
    for diz in dic_regioni[key_reg].keys():
        if dic_regioni[key_reg][diz] > 100:
            cc2.append(dic_regioni[key_reg][diz])
            cc.append(dic_regioni[key_reg][diz])
    # print(key_reg)
    # print(cc)
#    print(len(cc))
    # dic_2[key_reg] ={}
    dic_2[key_reg] = cc2
    if len(cc2) > 6:

        plt.plot(cc2[:5])
#        plt.hold(True)
#        leg.append(key_reg)
        print(key_reg + " parte da : "+  str(cc2[0]) + " e dopo 7 giorni " + str(cc2[5]) + " attualmente dopo " + str(len(cc2)) + " giorni, sono arrivati a " + str(cc2[len(cc2) - 1]))
#plt.legend(leg[0], leg[1], leg[2], leg[3], leg[4], leg[5], leg[6],leg[7], leg[8], leg[9], leg[10], leg[11], leg[12])
with open(r"C:\Users\massi\OneDrive\Desktop\coronavirus.json", 'w') as fp:
    
    json.dump(dic_regioni, fp, sort_keys=True, indent=4)

with open(r"C:\Users\massi\OneDrive\Desktop\coronavirus2.json", 'w') as fp:
    
    json.dump(dic_2, fp, sort_keys=True, indent=4)
