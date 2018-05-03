import matplotlib.pyplot as plt

with open("10_0.001.txt") as fp:
    lines = fp.readlines()
    x = []
    y = []
    for line in lines:
        tokens = line.split(",")
        x.append(int(tokens[0]))
        y.append(float(tokens[1]))
plt.plot(x[3:], y[3:], 'r--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
