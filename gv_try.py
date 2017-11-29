from graphviz import Digraph


def add_path(graph, path, color):
    for m in path[0]:
        graph.edge('L0'+'M'+str(m), 'sum_0', color=color)

    for i in range(1, len(path)-1):
        for m in path[i]:
            graph.edge('sum_'+str(i-1), 'L'+str(i)+'M'+str(m), color=color)

        for m in path[i]:
            graph.edge('L' + str(i) + 'M' + str(m), 'sum_'+str(i), color=color)

    for m in path[-1]:
        graph.edge('sum_'+str(len(path)-2), 'L'+str(len(path)-1)+'M'+str(m), color=color)
        graph.edge('L'+str(len(path)-1)+'M'+str(m),'sum_'+str(len(path)-1),  color=color)


def get_node_labels(depth, width):
    labels      = [[None]*width for _ in range(depth)]
    labels_t    = [[None]*depth for _ in range(width)]

    for m in range(width):
        for l in range(depth):
            labels[l][m]    = 'L'+str(l)+'M'+str(m)
            labels_t[m][l]  = 'L'+str(l)+'M'+str(m)

    return labels, labels_t


def create_color_map(depth, width, paths, colors, default_color='lightgray'):

    color_map = [[default_color for _ in range(width)] for _ in range(depth)]

    for path, color in zip(paths, colors):
        for l, layer in enumerate(path):
            for m in layer:
                if color_map[l][m] == default_color:
                    color_map[l][m] = color

    return color_map


def add_nodes(graph, labels, color_map):
    for l in range(len(labels)):
        for m in range(len(labels[l])):
            graph.node(labels[l][m], labels[l][m], color=color_map[l][m])



width = 10
depth = 3
labels, labels_t = get_node_labels(depth, width)
sums = ['sum_'+str(i) for i in range(depth)]

paths = [[[1], [0, 2], [1]],
         [[1], [1], [0, 2]]]
colors = ['lightblue', 'lightyellow3']

color_map = create_color_map(depth, width, paths, colors)


dot = Digraph(comment='example',
              graph_attr={'rank':'same', 'nodesep':'0.5', 'splines':'line'},
              edge_attr={'dir':'none'},
              node_attr={'style':'filled'})

add_nodes(dot, labels, color_map)

for s in sums:
    dot.node(s, label='sum', color='grey')

for i in range(len(paths)):
    add_path(dot, paths[i], colors[i])


dot.body += ['\t{rank = same; '+ ' '.join(row)+'}' for row in labels_t]
dot.body += ['\t{rank = same;'+' '.join(sums+['L0M'+str(int(width/2))])+'}']
dot.body += ['\t'+' -> '.join(column)+'[ style = invis, weight = 100 ];' for column in labels]

dot.view()