import plotly.graph_objects as go
    
def surface(x, y, fun, parallelize=True):
    assert x.shape == y.shape
    
    if parallelize:
        z = fun([x, y])
    else:
        z = []
        for i in range(len(x)):
            z.append([])
            for j in range(len(x[i])):
                z[i].append(fun([x[i][j], y[i][j]]))
    
    fig = go.Figure(
        data=[go.Surface(x=x, y=y, z=z)]
    )
    
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))

    return fig

def add_marker(pos, fig):
    fig.add_trace(
        go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode="markers",
            marker=dict(
                color="red",
                size=10,
            )
        )
    )
    
    return fig