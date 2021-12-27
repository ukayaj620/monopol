from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec


def create_image_dict(image, label, cmap):
    return dict({
        "image": image,
        "label": label,
        "cmap": cmap
    })


def show_plots(figure_size, ncols, nrows, images=[]):
    figure = plt.figure(figsize=figure_size)
    plt.rcParams.update({"font.size": 14})
    grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=figure)

    for i in range(len(images)):
        figure.add_subplot(grid[i])
        plt.axis(False)
        plt.title(images[i]["label"])
        plt.imshow(images[i]["image"], cmap=images[i]["cmap"])

    plt.show()
