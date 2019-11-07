import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#class Plot(object):
def show(data, throughput, it, path):
    fig = plt.figure(figsize=(20,5))
    plt.suptitle("% 12.2f" % (100*throughput), fontsize=64)
    data = list(map(list, zip(*data)))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1 , 1),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
    grid[0].imshow(data)  # The AxesGrid object work as a list of axes.
    plt.savefig(path + str(it) + ".png" , bbox_inches="tight")