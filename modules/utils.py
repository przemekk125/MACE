from torch_geometric.utils import scatter


def get_vectors_and_lengths():
    pass


def neighbour_sum(message, edge_index, num_nodes = None):
    receivers = edge_index[1]
    return scatter(src=message,index=receivers,reduce="sum", dim_size=num_nodes)