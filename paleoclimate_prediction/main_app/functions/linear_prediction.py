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


def get_plot(x: list, y: list):
    plt.switch_backend('AGG')
    plt.figure(figsize=(3, 3))
    plt.title('Prediction')
    plt.plot(x, y)
    plt.tight_layout()
    graph = get_graph()
    return graph


def linear_prediction():
    x = 2.0
    y = x + 2.0
    return y
