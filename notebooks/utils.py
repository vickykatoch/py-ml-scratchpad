from matplotlib import pyplot as plt

def plotData(x, y, h):
    plt.subplots()
    plt.plot(x, h)
    plt.scatter(x, y, c='red')
    plt.grid(color='gray', linestyle='--', linewidth=.6, axis='both', which='both', alpha=.4)
    plt.show()
