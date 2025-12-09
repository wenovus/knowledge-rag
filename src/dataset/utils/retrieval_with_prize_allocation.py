import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import functools
from .retrieval_func_selector import PrizeAllocation


def run_pcst(graph, n_prizes, q_emb, topk_e=3, cost_e=0.5):
    """
    Run PCST algorithm to select nodes and edges based on prizes and costs.

    Args:
        graph: PyTorch Geometric Data object
        n_prizes: Tensor of node prizes (shape: [num_nodes])
        q_emb: Query embedding tensor (for computing edge prizes)
        topk_e: Number of top edges to consider
        cost_e: Edge cost parameter

    Returns:
        selected_nodes: Array of selected node indices
        selected_edges: Array of selected edge indices
    """
    c = 0.01
    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    # Set up edge prizes
    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e_clamped = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e_clamped, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e_clamped
        for k in range(topk_e_clamped):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e_clamped-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges, device=n_prizes.device)

    # Build PCST input
    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.cpu().numpy()):
        prize_e = e_prizes[i].item()
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes.cpu().numpy(), np.array(vritual_n_prizes)])
    num_edges_original_graph = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges_original_graph]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = selected_edges + virtual_edges
    selected_edges = np.array(selected_edges)

    edge_index_selected = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index_selected[0].cpu().numpy(), edge_index_selected[1].cpu().numpy()]))

    return selected_nodes, selected_edges


def _allocate_prizes(node_similarities, topk_indices, prize_allocation, device):
    """
    Allocate prizes to top-k nodes based on prize_allocation mode.
    
    Args:
        node_similarities: Tensor of node similarities (shape: [num_nodes])
        topk_indices: Tensor of top-k node indices
        prize_allocation: PrizeAllocation enum value
        device: Device to create tensors on
    
    Returns:
        n_prizes: Tensor of node prizes (shape: [num_nodes])
    """
    num_nodes = node_similarities.size(0)
    topk_for_prize = topk_indices.size(0)
    n_prizes = torch.zeros(num_nodes, device=device)
    
    if prize_allocation == PrizeAllocation.LINEAR:
        # Linearly-proportional weights: rank 1 gets weight k, rank k gets weight 1
        prize_values = torch.arange(topk_for_prize, 0, -1, dtype=torch.float32, device=device)
    elif prize_allocation == PrizeAllocation.EQUAL:
        # Equal weighting for all top-k nodes
        prize_values = torch.ones(topk_for_prize, dtype=torch.float32, device=device)
    elif prize_allocation == PrizeAllocation.EXPONENTIAL:
        # Exponentially-decaying weights
        ranks = torch.arange(topk_for_prize, dtype=torch.float32, device=device)
        prize_values = torch.exp(-0.5 * ranks)
    else:
        raise ValueError(f"Unknown prize_allocation: {prize_allocation}. Must be one of {[mode.value for mode in PrizeAllocation]}")
    
    n_prizes[topk_indices] = prize_values
    return n_prizes


def retrieval_via_pcst_fn(prize_allocation):
    return functools.partial(retrieval_via_pcst, prize_allocation=prize_allocation)


def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, prize_allocation, topk=3, topk_e=3, cost_e=0.5):
    """
    PCST-based retrieval for subgraph extraction.

    Args:
        graph: PyTorch Geometric Data object
        q_emb: Query embedding tensor
        textual_nodes: DataFrame with node attributes
        textual_edges: DataFrame with edge attributes
        topk: Number of top nodes to select based on similarity
        topk_e: Number of top edges to consider
        cost_e: Edge cost parameter
        prize_allocation: Prize allocation mode (required, no default). Must be PrizeAllocation enum.

    Returns:
        data: Subgraph as PyTorch Geometric Data object
        desc: Description string with node and edge information
    """
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Set up node prizes
    device = graph.x.device
    node_similarities = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
    
    if topk > 0:
        topk_clamped = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(node_similarities, topk_clamped, largest=True)
        n_prizes = _allocate_prizes(node_similarities, topk_n_indices, prize_allocation, device)
    else:
        n_prizes = torch.zeros(graph.num_nodes, device=device)

    # Run PCST
    selected_nodes, selected_edges = run_pcst(graph, n_prizes, q_emb, topk_e, cost_e)

    edge_index = graph.edge_index[:, selected_edges]

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc
