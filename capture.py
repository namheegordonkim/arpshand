import numpy as np

from mocca_utils.plots.visplot import Figure, ArrowPlot

fig = Figure()

num_points = 1000
red = [1, 0, 0, 1]
green = [0, 1, 0, 1]

plot1 = ArrowPlot(
    figure=fig,
    xlim=[-20, 20],
    ylim=[-20, 20],
    plot_options={
        "width": 3,
        "arrow_size": 15,
        "arrow_color": [red],
        "color": np.repeat([red], num_points, axis=0),
    },
)

t = np.linspace(0, 2*np.pi, num_points).reshape(num_points, 1)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

mystery1 = np.expand_dims(np.concatenate((x, y), axis=1), axis=0)

plot2 = ArrowPlot(
    figure=None,
    plot_options={
        "parent": plot1,
        "width": 3,
        "arrow_size": 15,
        "arrow_color": [green],
        "color": np.repeat([green], num_points, axis=0),
    },
)

t = np.linspace(0, 12*np.pi, num_points).reshape(num_points, 1)
x = 2 * np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4*t) - np.sin(t/12) ** 5)
y = 2 * np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4*t) - np.sin(t/12) ** 5)

mystery2 = np.expand_dims(np.concatenate((x, y), axis=1), axis=0)
arrow2 = np.concatenate((mystery2[:, -1, :], mystery2[:, -2, :]), axis=1)
plot2.update(mystery2, arrow2)

while True:
    mystery1 = np.roll(mystery1, 1, axis=1)
    arrow1 = np.concatenate((mystery1[:, -1, :], mystery1[:, -2, :]), axis=1)
    plot1.update(mystery1, arrow1)
    fig.redraw()