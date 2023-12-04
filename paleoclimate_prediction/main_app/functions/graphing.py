import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# SOURCE: https://www.youtube.com/watch?v=jrT6NiM46jk&t=98s


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph


def get_bar_plot(names: list, values1: list, values2: list):
    plt.switch_backend('AGG')
    plt.style.use("ggplot")
    plt.figure(figsize=(3, 3))
    axes = plt.axes()
    axes.set_ylim([0, 1])
    plt.title('MAT Prediction')

    x_indexes = np.arange(len(names))

    bar_width = 0.25
    plt.bar(x=x_indexes-bar_width/2, height=values1, width=bar_width, color="red", label="BHPO",)
    plt.bar(x=x_indexes+bar_width/2, height=values2, width=bar_width, color="blue", label="AHPO",)

    plt.xlabel("MODELS")
    plt.ylabel("SCORES")
    plt.xticks(ticks=x_indexes, labels=names)
    plt.tight_layout()
    graph = get_graph()
    return graph

