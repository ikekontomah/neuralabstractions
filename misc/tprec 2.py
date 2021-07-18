import plotly.graph_objects as go


tprec = [ 0.91, 0.773, 0.315, 0.174, 0.143, 0.147, 0.164, 0.018, 0.034, 0.1]
tprec_pen = [0.935, 0.792, 0.2615, 0.143, 0.24, 0.1056, 0.2116, 0.029, 0.067, 0.15]

progs = ['n=1', 'n=2', 'n=3', 'n=4', 'n=5', 'n=6', 'n=7', 'n=8', 'n=9', 'n=10']

fig = go.Figure()

fig.add_trace(go.Bar(
    x=progs,
    y=tprec,
    name = 'tprec-score',
    marker_color= 'indianred'
))
fig.add_trace(go.Bar(
    x=progs,
    y=tprec_pen,
    name='tprec-score with penalty',
    marker_color= 'lightsalmon'
))

fig.update_layout(title='Average tprec-scores for 18 human subjects',
                   xaxis_title='Length of sequence',
                   yaxis_title='Average tprec-score',
                   barmode='group',
                   bargap=0.1)

fig.show()