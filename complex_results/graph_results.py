import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

with open("en_results.txt") as fp:
    lines = fp.readlines()
    results = []
    for line in lines:
        token_dict = {}
        tokens = line.rstrip().split(",")
        for token in tokens:
            item = token.lstrip().split(":")
            token_dict[item[0]] = float(item[1])
        results.append(token_dict)
print results
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []
for item in results:
    print results
    if item["Hidden Layer Size"] == 5.0:
        print item["Learning Rate"]
        x1 += [item["Learning Rate"]]
        y1 += [item["Loss"]]
    elif item["Hidden Layer Size"] == 15.0:
        x2 += [item["Learning Rate"]]
        y2 += [item["Loss"]]
    elif item["Hidden Layer Size"] == 25.0:
        x3 += [item["Learning Rate"]]
        y3 += [item["Loss"]]
    elif item["Hidden Layer Size"] == 35.0:
        x4 += [item["Learning Rate"]]
        y4 += [item["Loss"]]
plt.plot(x1, y1, 'ro-', x2, y2, 'go-', x3, y3, 'bo-', x4, y4, 'yo-')
red_patch = mpatches.Patch(color='red', label='HL = 5')
green_patch = mpatches.Patch(color='green', label='HL = 15')
blue_patch = mpatches.Patch(color='blue', label='HL = 25')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()
