from matplotlib import pyplot as plt

def plotData(x, y, h, xlabel, ylabel, xlim, ylim):
    plt.subplots()
    plt.plot(x, h)
    plt.scatter(x, y, c='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(color='gray', linestyle='--', linewidth=.6,
             axis='both', which='both', alpha=.4)
    plt.show()