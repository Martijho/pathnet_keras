from graphviz import Digraph

class PathNetPlotter():
    def __init__(self, pathnet, paths=None):
        self._pathnet = pathnet

    def plot_paths(self, paths, filename=None):
        depth = self._pathnet.depth
        width = self._pathnet.width

        tc = self._pathnet.training_counter
        labels, labels_t = self.get_node_labels()
        sums = ['sum_' + str(i) for i in range(depth)]

        paths, colors = self.get_paths_to_plot(paths)
        color_map = self.create_color_map(paths, colors)

        dot = Digraph(comment='example',
                      graph_attr={'rank': 'same', 'nodesep': '0.5', 'splines': 'line'},
                      edge_attr={'dir': 'none', 'penwidth':'3'},
                      node_attr={'style': 'filled'})

        self.add_nodes(dot, labels, color_map, names=tc)

        for s in sums:
            dot.node(s, label='sum', color='grey')

        for i in range(len(paths)):
            self.add_path(dot, paths[i], colors[i])

        dot.body += ['\t{rank = same; ' + ' '.join(row) + '}' for row in labels_t]
        dot.body += ['\t{rank = same; ' + ' '.join(sums + ['L0M' + str(int(width / 2))]) + '}']
        dot.body += ['\t' + ' -> '.join(column) + '[ style = invis, weight = 100 ];' for column in labels]

        if filename is not None:
            dot.view()
            dot.render(filename, view=False)



    def add_path(self, graph, path, color):
        for m in path[0]:
            graph.edge('L0' + 'M' + str(m), 'sum_0', color=color)

        for i in range(1, len(path) - 1):
            for m in path[i]:
                graph.edge('sum_' + str(i - 1), 'L' + str(i) + 'M' + str(m), color=color)

            for m in path[i]:
                graph.edge('L' + str(i) + 'M' + str(m), 'sum_' + str(i), color=color)

        for m in path[-1]:
            graph.edge('sum_' + str(len(path) - 2), 'L' + str(len(path) - 1) + 'M' + str(m), color=color)
            graph.edge('L' + str(len(path) - 1) + 'M' + str(m), 'sum_' + str(len(path) - 1), color=color)

    def get_node_labels(self):
        depth = self._pathnet.depth
        width = self._pathnet.width
        labels = [[None] * width for _ in range(depth)]
        labels_t = [[None] * depth for _ in range(width)]

        for m in range(width):
            for l in range(depth):
                labels[l][m] = 'L' + str(l) + 'M' + str(m)
                labels_t[m][l] = 'L' + str(l) + 'M' + str(m)

        return labels, labels_t

    def create_color_map(self, paths, colors, default_color='lightgray'):
        depth = self._pathnet.depth
        width = self._pathnet.width
        color_map = [[default_color for _ in range(width)] for _ in range(depth)]

        for path, color in zip(paths, colors):
            for l, layer in enumerate(path):
                for m in layer:
                    if color_map[l][m] == default_color:
                        color_map[l][m] = color

        return color_map

    def add_nodes(self, graph, labels, color_map, names=None):
        if names is None:
            names = labels

        for l in range(len(labels)):
            for m in range(len(labels[l])):
                graph.node(labels[l][m], str(names[l][m]), color=color_map[l][m])

    def get_paths_to_plot(self, paths):

        if paths is None:
            paths = []
            for task in self._pathnet._tasks:
                if task.optimal_path is not None:
                    paths.append(task.optimal_path)

        colors = ['#740040', '#df5454', '#fb9e3f',
                  '#8ed8b8', '#d3c8ee']

        return paths, colors[:len(paths)]