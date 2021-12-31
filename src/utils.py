from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np
import io
import cv2


def create_image_dict(image, label, cmap):
    return dict({
        "image": image,
        "label": label,
        "cmap": cmap
    })


def get_image_from_figure(figure, dpi=180):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    image_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()
    image = cv2.imdecode(image_array, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_plots(figure_size, ncols, nrows, images=[], font_size=14):
    figure = plt.figure(figsize=figure_size)
    plt.ion()
    plt.rcParams.update({"font.size": font_size})
    grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=figure)

    for i in range(len(images)):
        figure.add_subplot(grid[i])
        plt.axis(False)
        plt.title(images[i]["label"])
        plt.imshow(images[i]["image"], cmap=images[i]["cmap"])

    plt.close(figure)
    plt.ioff()
    return get_image_from_figure(figure)


def show_plots(figure_size, ncols, nrows, images=[], font_size=14):
    figure = plt.figure(figsize=figure_size)
    plt.rcParams.update({"font.size": font_size})
    grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=figure)

    for i in range(len(images)):
        figure.add_subplot(grid[i])
        plt.axis(False)
        plt.title(images[i]["label"])
        plt.imshow(images[i]["image"], cmap=images[i]["cmap"])

    plt.show()
