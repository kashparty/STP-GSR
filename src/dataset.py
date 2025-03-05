import torch
import numpy as np
import networkx as nx
from functools import partial
from torch_geometric.data import Data
from src.matrix_vectorizer import MatrixVectorizer


def load_dataset(config):
    n_source_nodes = config.dataset.n_source_nodes
    n_target_nodes = config.dataset.n_target_nodes
    n_samples = config.dataset.n_samples

    # Generate Erdos-Renyi graphs
    if config.dataset.name == 'er':
        # Probability of an edge between two nodes
        source_edge_prob = config.dataset.source_edge_prob
        target_edge_prob = config.dataset.target_edge_prob

        source_mat_all = [create_er_graph(n_source_nodes, source_edge_prob) for _ in range(n_samples)]
        target_mat_all = [create_er_graph(n_target_nodes, target_edge_prob) for _ in range(n_samples)]

    # Generate Barabasi-Albert graphs
    elif config.dataset.name == 'ba':
        # Number of edges to attach from a new node to existing nodes
        n_source_edges_per_node = config.dataset.n_source_edges_per_node
        n_target_edges_per_node = config.dataset.n_target_edges_per_node

        source_mat_all = [create_ba_graph(n_source_nodes, n_source_edges_per_node) for _ in range(n_samples)]
        target_mat_all = [create_ba_graph(n_target_nodes, n_target_edges_per_node) for _ in range(n_samples)]

    # Generate Kronecker graphs
    elif config.dataset.name == 'kronecker':
        # Initial graph template size
        source_init_matrix_size = config.dataset.source_init_matrix_size
        target_init_matrix_size = config.dataset.target_init_matrix_size
        
        # Number of times Kronecker product is applied to exponentially expand the graph
        n_iterations = config.dataset.n_iterations

        assert n_source_nodes == source_init_matrix_size ** n_iterations, "Number of nodes in source graph does not match size of Kronecker graph"
        assert n_target_nodes == target_init_matrix_size ** n_iterations, "Number of nodes in target graph does not match size of Kronecker graph"

        source_mat_all = [create_kronecker_graph(source_init_matrix_size, n_iterations) for _ in range(n_samples)]
        target_mat_all = [create_kronecker_graph(target_init_matrix_size, n_iterations) for _ in range(n_samples)]

    # Generate SBM graphs
    elif config.dataset.name == 'sbm':
        # Number of blocks in each cluster of the SBM graph
        source_blocks = config.dataset.source_blocks
        target_blocks = config.dataset.target_blocks

        assert n_source_nodes == sum(source_blocks), "Number of nodes in source graph does not match sum of nodes in blocks"
        assert n_target_nodes == sum(target_blocks), "Number of nodes in target graph does not match sum of nodes in blocks"

        # Probability matrix for edges between blocks
        source_P = np.array(config.dataset.source_P)
        target_P = np.array(config.dataset.target_P)

        source_mat_all = [create_sbm_graph(source_blocks, source_P) for _ in range(n_samples)]
        target_mat_all = [create_sbm_graph(target_blocks, target_P) for _ in range(n_samples)]
    
    # Generate training graphs
    elif config.dataset.name == "train":
        source_vectorized = np.genfromtxt("lr_train.csv", delimiter=",", skip_header=1)
        target_vectorized = np.genfromtxt("hr_train.csv", delimiter=",", skip_header=1)

        source_mat_all = [MatrixVectorizer.anti_vectorize(source_vectorized, 160)]
        target_mat_all = [MatrixVectorizer.anti_vectorize(target_vectorized, 268)]

    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset.name}")
    
    # Convert to torch tensors
    source_mat_all = [torch.tensor(x, dtype=torch.float) for x in source_mat_all]
    target_mat_all = [torch.tensor(x, dtype=torch.float) for x in target_mat_all]
        
    # Convert to PyG
    node_feat_init = config.dataset.node_feat_init
    node_feat_dim = config.dataset.node_feat_dim
    pyg_partial = partial(create_pyg_graph, node_feature_init=node_feat_init, node_feat_dim=node_feat_dim)
    
    source_pyg_all = [pyg_partial(x, n_source_nodes) for x in source_mat_all]
    target_pyg_all = [pyg_partial(x, n_target_nodes) for x in target_mat_all]

    # Prepare source and target data
    source_data = [{'pyg': source_pyg, 'mat': source_mat} for source_pyg, source_mat in zip(source_pyg_all, source_mat_all)]
    target_data = [{'pyg': target_pyg, 'mat': target_mat} for target_pyg, target_mat in zip(target_pyg_all, target_mat_all)]

    return source_data, target_data


def create_er_graph(n_nodes, edge_prob):
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    adj = nx.adjacency_matrix(G).toarray()
    
    return adj


def create_ba_graph(n_nodes, n_edges):
    G = nx.barabasi_albert_graph(n_nodes, n_edges)
    adj = nx.adjacency_matrix(G).toarray()
    
    return adj


def create_symmetric_initiator_matrix(size, low=0.5, high=1.0, diagonal_value=0.1):
    """
    Creates a symmetric initiator matrix with values uniformly generated from [low, high].

    Args:
        size (int): The size of the initiator matrix (size x size).
        low (float): The lower bound of the uniform distribution.
        high (float): The upper bound of the uniform distribution.

    Returns:
        np.array: A symmetric initiator matrix of the specified size.
    """
    # Generate a random matrix with values uniformly distributed in [low, high]
    matrix = np.random.uniform(low, high, (size, size))
    
    # Make the matrix symmetric
    symmetric_matrix = (matrix + matrix.T) / 2

    # Set diagonal entries to the specified value
    np.fill_diagonal(symmetric_matrix, diagonal_value)
    
    return symmetric_matrix


def kronecker_product(init_matrix, iterations):
    """
    Generate a Kronecker graph adjacency matrix by recursively applying the Kronecker product.

    Args:
        init_matrix (np.array): Initiator matrix
        iterations (int): Number of iterations

    Returns:
        np.array: Adjacency matrix of the Kronecker graph
    """
    result = init_matrix
    for _ in range(iterations - 1):
        result = np.kron(result, init_matrix)
    return result


def create_kronecker_graph(init_matrix_size, iterations):
    """
    Generates a Kronecker graph using an initiator matrix and a specified number of iterations.

    Args:
        init_matrix (np.array): Initiator matrix
        iterations (int): Number of iterations to apply the Kronecker product

    Returns:
        G (networkx.Graph): Generated Kronecker graph
    """
    init_matrix = create_symmetric_initiator_matrix(init_matrix_size, 0.5, 1.0)
    adj_matrix = kronecker_product(init_matrix, iterations)
    G = nx.from_numpy_array((adj_matrix > np.random.rand(*adj_matrix.shape)).astype(int))

    # Ensure the graph is connected
    if not nx.is_connected(G):
        # Add edges to make the graph connected
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)

    adj = nx.adjacency_matrix(G).toarray()

    return adj


def create_sbm_graph(block_sizes, P):
    """
    Generate a stochastic block model (SBM) graph.

    Parameters:
    block_sizes : list of int
        Sizes of each block.
    P : np.ndarray
        Edge probability matrix.

    Returns:
    G : nx.Graph
        The generated SBM graph.
    """
    # Number of nodes
    N = sum(block_sizes)
    
    # Initialize the graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    # Create block membership for each node
    block_membership = []
    current_node = 0
    for block_id, size in enumerate(block_sizes):
        block_membership.extend([block_id] * size)
        current_node += size
    
    # Generate edges based on block memberships and probabilities in P
    for i in range(N):
        for j in range(i + 1, N):
            block_i = block_membership[i]
            block_j = block_membership[j]
            prob_edge = P[block_i, block_j]
            
            if np.random.rand() < prob_edge:
                G.add_edge(i, j)

    adj = nx.adjacency_matrix(G).toarray()
    
    return adj


def create_pyg_graph(x, n_nodes, node_feature_init='adj', node_feat_dim=1):
    """
    Create a PyTorch Geometric graph data object from given adjacency matrix.
    """
    # Initialise edge features
    if isinstance(x, torch.Tensor):
        edge_attr = x.view(-1, 1)
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)


    # Initialise node features
    # From adjacency matrix
    if node_feature_init == 'adj':
        if isinstance(x, torch.Tensor):
            # node_feat = x.clone().detach()
            node_feat = x
        else:
            node_feat = torch.tensor(x, dtype=torch.float)

    # Random initialisation
    elif node_feature_init == 'random':
        node_feat = torch.randn(n_nodes, node_feat_dim, device=edge_attr.device)

    # Ones initialisation
    elif node_feature_init == 'ones':
        node_feat = torch.ones(n_nodes, node_feat_dim, device=edge_attr.device)

    else:
        raise ValueError(f"Unsupported node feature initialization: {node_feature_init}")


    rows, cols = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    pyg_graph = Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
    
    return pyg_graph
