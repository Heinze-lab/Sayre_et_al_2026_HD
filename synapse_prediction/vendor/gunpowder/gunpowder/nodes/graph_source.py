import logging
import networkx as nx
import numpy as np

from gunpowder.batch import Batch
from gunpowder.graph import Graph
from gunpowder.graph_spec import GraphSpec
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.profiling import Timing
from funlib.persistence import graphs as grs

logger = logging.getLogger(__name__)


class GraphSource(BatchProvider):
    """Creates a gunpowder graph source from a daisy graph provider.
    Queries for graphs from a given Roi will only return edges completely
    contained within the Roi - edges that cross the boundary will not be
    included.

    Arguments:

        graph_provider (:class:`daisy.SharedGraphProvider`):
            A daisy graph provider to read the graph from.
            Can be backed by MongoDB or any other implemented backend.

        graph (:class:`GraphKey`):
            The key of the graph to create

        graph_spec (:class:`GraphSpec`, optional):
            An optional :class:`GraphSpec` containing a roi and optionally
            whether the graph is directed. The default is to have an unbounded
            roi and detect directedness from the graph_provider.
    """

    def __init__(self, graph_params, graph, graph_spec=None):
        [db_name, db_host] = graph_params
        rag_provider = grs.MongoDbGraphProvider(
         db_name,
         host = db_host,
         mode = 'r+',
         directed = True,
         edges_collection = 'edges_pre_post_syn',
         position_attribute = ['center_z', 'center_y', 'center_x'])
        self.graph_provider = rag_provider
        self.graph = graph
        self.graph_spec = graph_spec

    def setup(self):
        if self.graph_spec is not None:
            roi = self.graph_spec.roi
            #print(self.graph_provider.spec)
            if self.graph_spec.directed is not None:
                assert self.graph_spec.directed == self.graph_provider.directed
        else:
            roi = None
        spec = GraphSpec(roi=roi, directed=self.graph_provider.directed)
        self.provides(self.graph, spec)

    def provide(self, request):
        timing = Timing(self)
        timing.start()
        batch = Batch()
        roi = request[self.graph].roi.copy()
        #print(self.graph_provider)
        #print(roi)
        graph = GraphSource.create_gp_graph_from_daisy(self.graph_provider, roi)
        batch.graphs[self.graph] = graph

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    @staticmethod
    def create_gp_graph_from_daisy(graph_provider, roi):
        """A static method to convert a daisy graph into a gunpowder graph.
        Only includes edges if both endpoints are within the roi.

        Arguments:
            graph_provider (:class:`daisy.SharedGraphProvider`)
                A daisy graph provider to read the graph from

            roi (:class:`Roi`):
                The roi in which to read the graph

        Returns:
            An instance of :class:`Graph` containing the nodes and edges read
            from the daisy graph provider in the given roi.

        """
        logger.debug("Creating gunpowder graph from daisy graph provider")
        #print("GRAPH PROVIDE HERE")
        #print(graph_provider)
        #print('________')
        #print(roi)
        #print(type(graph_provider))
        #print('__________')
        daisy_graph = graph_provider[roi]
        logger.debug("%d nodes found in roi %s", len(daisy_graph), roi)
        spec = GraphSpec(roi=roi, directed=daisy_graph.is_directed())
        dangling_nodes = []
        for node, data in daisy_graph.nodes(data=True):
            position_attribute = graph_provider.position_attribute
            if type(position_attribute) == list:
                if position_attribute[0] not in data:
                    dangling_nodes.append(node)
                    continue
                location = np.array(
                    [data[attr] for attr in position_attribute], dtype=np.float32
                )
            else:
                if position_attribute not in data:
                    dangling_nodes.append(node)
                    continue
                location = np.array(data[position_attribute], dtype=np.float32)
            data["location"] = location
            data["id"] = node

        logger.debug("Dangling nodes: %s", dangling_nodes)
        for n in dangling_nodes:
            daisy_graph.remove_node(n)

        if daisy_graph.is_directed():
            pure_nx_graph = nx.DiGraph()
        else:
            pure_nx_graph = nx.Graph()

        pure_nx_graph.update(daisy_graph)
        return Graph.from_nx_graph(pure_nx_graph, spec)
