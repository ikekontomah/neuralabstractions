import plotly.graph_objects as go


ffwd = [ 9.0024, 9.1650, 8.6867, 8.8713, 9.4375, 8.9051]
cnn =  [ 9.1401, 9.0965, 9.1220,  8.9095, 8.8314, 9.5589]

progs = ['n=1', 'n=2', 'n=4', 'n=6', 'n=8', 'n=10']

fig = go.Figure()

fig.add_trace(go.Bar(
    x=ffwd,
    y=progs,
    orientation='h',
    name = 'Feedforward network',
    marker_color= 'indianred'
))
fig.add_trace(go.Bar(
    x=cnn,
    y=progs,
    orientation='h',
    name='Convolutional network',
    marker_color= 'lightsalmon'
))

fig.update_layout(title='Average Starting Perplexity of Feedforward and CNN Models',
                   xaxis_title='Average starting perplexity',
                   yaxis_title='Maximum sequence length',
                   barmode='group',
                   bargap=0.1)

fig.show()