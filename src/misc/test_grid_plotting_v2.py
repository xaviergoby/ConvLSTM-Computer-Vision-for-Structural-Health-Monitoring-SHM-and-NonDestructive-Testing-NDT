from matplotlib import pyplot as plt
from matplotlib import colors

data = [
    [0,0,0,0,0,1,1,1,1,0],
    [0,0,0,0,0,1,0,0,1,0],
    [0,0,1,0,1,0,1,1,0,0],
    [0,0,1,0,0,1,1,0,1,0],
    [0,0,1,0,1,0,0,1,1,0],
    [1,0,0,1,0,1,0,0,1,0],
    [0,1,0,0,0,1,1,1,1,1],
    [0,1,0,0,0,0,1,1,1,1],
    [1,0,0,0,1,1,1,0,1,0],
    [1,1,1,1,0,0,0,1,1,0]
]

cmap = colors.ListedColormap(['White','Black'])
plt.figure(figsize=(6,6))
plt.pcolor(data[::-1],cmap=cmap,edgecolors='k', linewidths=3)
plt.show()