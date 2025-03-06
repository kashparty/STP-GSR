import gc
import torch


def create_dual_graph(adjacency_matrix):
    """Returns edge_index and node_feature_matrix for the dual graph."""
    # Number of nodes in the original graph G1
    n = adjacency_matrix.shape[0]

    # Find all potential edges in the upper triangular part
    row, col = torch.triu_indices(n, n, offset=1, device=adjacency_matrix.device)
    all_edges = torch.stack([row, col], dim=1)
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1)

    # Filter actual edges
    actual_edges = all_edges[actual_edges_mask]

    # Number of edges in G1
    num_actual_edges = actual_edges.shape[0]
    max_possible_edges = row.size(0)

    # Create a tensor indicating shared nodes between edges (Incidence matrix)
    edge_to_nodes = torch.zeros((max_possible_edges, n), dtype=torch.float, device=adjacency_matrix.device)
    edge_to_nodes[actual_edges_mask, actual_edges[:, 0]] = 1.0
    edge_to_nodes[actual_edges_mask, actual_edges[:, 1]] = 1.0

    # Compute the connectivity between edges
    shared_nodes_matrix = edge_to_nodes @ edge_to_nodes.t()
    shared_nodes_matrix.fill_diagonal_(0)  # Remove self-loops

    # Extract edge indices from the connectivity matrix
    edge_index = shared_nodes_matrix.nonzero(as_tuple=False).t().contiguous()

    # Create node feature matrix for the dual graph
    node_feat_matrix = torch.zeros((max_possible_edges, 1), dtype=torch.float, device=adjacency_matrix.device)
    node_feat_matrix[actual_edges_mask] = adjacency_matrix[actual_edges[:, 0], actual_edges[:, 1]].view(-1, 1).float()

    torch.cuda.empty_cache()
    gc.collect()

    return edge_index, node_feat_matrix


def revert_dual(node_feat, n_nodes):
    """Reverts the dual node feature matrix to the original adjacency matrix."""
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float, device=node_feat.device)
    row, col = torch.triu_indices(n_nodes, n_nodes, offset=1)
    adj[row, col] = node_feat.view(-1)
    adj[col, row] = node_feat.view(-1)

    torch.cuda.empty_cache()
    gc.collect()
    
    return adj


def create_dual_graph_feature_matrix(adjacency_matrix):
    """Returns node_feature_matrix for the dual graph."""
    # Number of nodes in the original graph G1
    n = adjacency_matrix.shape[0]

    # Find all potential edges in the upper triangular part
    row, col = torch.triu_indices(n, n, offset=1, device=adjacency_matrix.device)
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1)

    # Create node feature matrix for the dual graph
    node_feat_matrix = torch.zeros((row.size(0), 1), dtype=torch.float, device=adjacency_matrix.device)
    node_feat_matrix[actual_edges_mask] = adjacency_matrix[row[actual_edges_mask], col[actual_edges_mask]].view(-1, 1).float()

    return node_feat_matrix
