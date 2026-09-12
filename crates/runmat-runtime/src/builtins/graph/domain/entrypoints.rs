use super::algorithms::{
    adjacency_lists, adjacency_tensor, assign_tree_positions, degree_counts, dijkstra,
    edge_matches, indegree_counts, neighbor_query, neighbors_for, node_list_value,
    node_metric_value, outdegree_counts, reconstruct_path, tensor_value, NeighborMode,
};
use super::construction::{construct_graph, graph_data};
use super::parsing::{node_index_from_value, node_indices_from_value, numeric_vector};
use super::*;

#[runtime_builtin(
    name = "graph",
    category = "graph",
    summary = "Create an undirected graph object.",
    keywords = "graph,edges,nodes,network",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn graph_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    construct_graph(GRAPH_NAME, false, gather_values(args).await?)
}

#[runtime_builtin(
    name = "digraph",
    category = "graph",
    summary = "Create a directed graph object.",
    keywords = "digraph,directed graph,edges,nodes,network",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::DIGRAPH_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn digraph_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    construct_graph(DIGRAPH_NAME, true, gather_values(args).await?)
}

#[runtime_builtin(
    name = "numnodes",
    category = "graph",
    summary = "Return the number of nodes in a graph.",
    keywords = "numnodes,graph,nodes",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn numnodes_builtin(graph: Value) -> BuiltinResult<Value> {
    Ok(Value::Num(
        graph_data(&graph, "numnodes")?.node_count as f64,
    ))
}

#[runtime_builtin(
    name = "numedges",
    category = "graph",
    summary = "Return the number of edges in a graph.",
    keywords = "numedges,graph,edges",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn numedges_builtin(graph: Value) -> BuiltinResult<Value> {
    Ok(Value::Num(
        graph_data(&graph, "numedges")?.edges.len() as f64
    ))
}

#[runtime_builtin(
    name = "adjacency",
    category = "graph",
    summary = "Return a dense adjacency matrix for a graph.",
    keywords = "adjacency,graph,matrix",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn adjacency_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "adjacency")?;
    Ok(Value::Tensor(adjacency_tensor(&graph, "adjacency")?))
}

#[runtime_builtin(
    name = "degree",
    category = "graph",
    summary = "Return graph node degree counts.",
    keywords = "degree,graph,nodes",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn degree_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "degree")?;
    node_metric_value(degree_counts(&graph), "degree")
}

#[runtime_builtin(
    name = "indegree",
    category = "graph",
    summary = "Return node indegree counts.",
    keywords = "indegree,digraph,graph,nodes",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn indegree_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "indegree")?;
    node_metric_value(indegree_counts(&graph), "indegree")
}

#[runtime_builtin(
    name = "outdegree",
    category = "graph",
    summary = "Return node outdegree counts.",
    keywords = "outdegree,digraph,graph,nodes",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn outdegree_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "outdegree")?;
    node_metric_value(outdegree_counts(&graph), "outdegree")
}

#[runtime_builtin(
    name = "neighbors",
    category = "graph",
    summary = "Return neighboring graph nodes.",
    keywords = "neighbors,graph,nodes",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn neighbors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
    neighbor_query(graph, node, NeighborMode::Undirected, "neighbors").await
}

#[runtime_builtin(
    name = "successors",
    category = "graph",
    summary = "Return successor nodes in a directed graph.",
    keywords = "successors,digraph,graph,nodes",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn successors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
    neighbor_query(graph, node, NeighborMode::Successors, "successors").await
}

#[runtime_builtin(
    name = "predecessors",
    category = "graph",
    summary = "Return predecessor nodes in a directed graph.",
    keywords = "predecessors,digraph,graph,nodes",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn predecessors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
    neighbor_query(graph, node, NeighborMode::Predecessors, "predecessors").await
}

#[runtime_builtin(
    name = "findedge",
    category = "graph",
    summary = "Find edge indices between graph nodes.",
    keywords = "findedge,graph,edges",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn findedge_builtin(
    graph: Value,
    source: Value,
    target: Value,
) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "findedge")?;
    let source = node_index_from_value(&graph, &source, "findedge")?;
    let target = node_index_from_value(&graph, &target, "findedge")?;
    for (idx, edge) in graph.edges.iter().enumerate() {
        if edge_matches(edge, source, target, graph.directed) {
            return Ok(Value::Num((idx + 1) as f64));
        }
    }
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "shortestpath",
    category = "graph",
    summary = "Compute a shortest path between graph nodes.",
    keywords = "shortestpath,graph,path,dijkstra",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn shortestpath_builtin(
    graph: Value,
    source: Value,
    target: Value,
) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "shortestpath")?;
    let source = node_index_from_value(&graph, &source, "shortestpath")?;
    let target = node_index_from_value(&graph, &target, "shortestpath")?;
    let (distances, previous) = dijkstra(&graph, source);
    let path = reconstruct_path(source, target, &previous);
    let path_value = node_list_value(&graph, &path, "shortestpath")?;
    let distance = distances[target];
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![path_value])),
        Some(2) => Ok(Value::OutputList(vec![path_value, Value::Num(distance)])),
        Some(_) => Err(graph_error(
            "shortestpath",
            "shortestpath: too many output arguments; maximum is 2",
        )),
        None => Ok(path_value),
    }
}

#[runtime_builtin(
    name = "distances",
    category = "graph",
    summary = "Compute shortest-path distances between graph nodes.",
    keywords = "distances,graph,dijkstra",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn distances_builtin(graph: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "distances")?;
    let rest = gather_values(rest).await?;
    let sources = match rest.first() {
        Some(value) => node_indices_from_value(&graph, value, "distances")?,
        None => (0..graph.node_count).collect(),
    };
    let targets = match rest.get(1) {
        Some(value) => node_indices_from_value(&graph, value, "distances")?,
        None => (0..graph.node_count).collect(),
    };
    let mut data = vec![0.0; sources.len() * targets.len()];
    for (source_row, source) in sources.iter().copied().enumerate() {
        let (distances, _) = dijkstra(&graph, source);
        for (target_col, target) in targets.iter().copied().enumerate() {
            data[source_row + target_col * sources.len()] = distances[target];
        }
    }
    tensor_value(data, vec![sources.len(), targets.len()], "distances")
}

#[runtime_builtin(
    name = "conncomp",
    category = "graph",
    summary = "Return weak connected components for graph nodes.",
    keywords = "conncomp,graph,components",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn conncomp_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "conncomp")?;
    let mut bins = vec![0.0; graph.node_count];
    let mut component = 0usize;
    for start in 0..graph.node_count {
        if bins[start] != 0.0 {
            continue;
        }
        component += 1;
        let mut queue = VecDeque::from([start]);
        bins[start] = component as f64;
        while let Some(node) = queue.pop_front() {
            for next in neighbors_for(&graph, node, NeighborMode::Undirected) {
                if bins[next] == 0.0 {
                    bins[next] = component as f64;
                    queue.push_back(next);
                }
            }
        }
    }
    tensor_value(bins, vec![1, graph.node_count], "conncomp")
}

#[runtime_builtin(
    name = "toposort",
    category = "graph",
    summary = "Return a topological ordering of a directed acyclic graph.",
    keywords = "toposort,digraph,directed acyclic graph",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn toposort_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "toposort")?;
    if !graph.directed {
        return Err(graph_error(
            "toposort",
            "toposort: expected a digraph input",
        ));
    }
    let mut indegree = vec![0usize; graph.node_count];
    let adj = adjacency_lists(&graph, NeighborMode::Successors);
    for edge in &graph.edges {
        indegree[edge.target] += 1;
    }
    let mut queue = VecDeque::new();
    for (idx, degree) in indegree.iter().copied().enumerate() {
        if degree == 0 {
            queue.push_back(idx);
        }
    }
    let mut order = Vec::with_capacity(graph.node_count);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &next in &adj[node] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push_back(next);
            }
        }
    }
    if order.len() != graph.node_count {
        return Err(graph_error("toposort", "toposort: graph contains a cycle"));
    }
    node_list_value(&graph, &order, "toposort")
}

#[runtime_builtin(
    name = "bfsearch",
    category = "graph",
    summary = "Return breadth-first search node order.",
    keywords = "bfsearch,graph,breadth first search",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn bfsearch_builtin(graph: Value, start: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "bfsearch")?;
    let start = node_index_from_value(&graph, &start, "bfsearch")?;
    let mut visited = vec![false; graph.node_count];
    let mut queue = VecDeque::from([start]);
    let mut order = Vec::new();
    visited[start] = true;
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for next in neighbors_for(&graph, node, NeighborMode::Successors) {
            if !visited[next] {
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }
    node_list_value(&graph, &order, "bfsearch")
}

#[runtime_builtin(
    name = "dfsearch",
    category = "graph",
    summary = "Return depth-first search node order.",
    keywords = "dfsearch,graph,depth first search",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn dfsearch_builtin(graph: Value, start: Value) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, "dfsearch")?;
    let start = node_index_from_value(&graph, &start, "dfsearch")?;
    let adj = adjacency_lists(&graph, NeighborMode::Successors);
    let mut visited = vec![false; graph.node_count];
    let mut stack = vec![start];
    let mut order = Vec::new();
    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        order.push(node);
        for &next in adj[node].iter().rev() {
            if !visited[next] {
                stack.push(next);
            }
        }
    }
    node_list_value(&graph, &order, "dfsearch")
}

#[runtime_builtin(
    name = "treelayout",
    category = "graph",
    summary = "Compute simple tree-layout coordinates from a parent vector.",
    keywords = "treelayout,tree,layout,graph",
    type_resolver(numeric_type),
    descriptor(crate::builtins::graph::GRAPH_QUERY_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn treelayout_builtin(parents: Value) -> BuiltinResult<Value> {
    let parents = gather_if_needed_async(&parents).await?;
    let parents = numeric_vector(&parents, "treelayout")?;
    if parents.is_empty() {
        return Ok(Value::OutputList(vec![
            tensor_value(Vec::new(), vec![1, 0], "treelayout")?,
            tensor_value(Vec::new(), vec![1, 0], "treelayout")?,
        ]));
    }
    let n = parents.len();
    let mut children = vec![Vec::<usize>::new(); n];
    let mut roots = Vec::new();
    for (idx, parent) in parents.iter().copied().enumerate() {
        if parent == 0 {
            roots.push(idx);
        } else if parent <= n {
            children[parent - 1].push(idx);
        } else {
            return Err(graph_error(
                "treelayout",
                "treelayout: parent indices must be zero or valid one-based node ids",
            ));
        }
    }
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut leaf_cursor = 1.0;
    for root in roots {
        assign_tree_positions(root, 0.0, &children, &mut leaf_cursor, &mut x, &mut y);
    }
    let x_value = tensor_value(x, vec![1, n], "treelayout")?;
    let y_value = tensor_value(y, vec![1, n], "treelayout")?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![x_value])),
        Some(2) => Ok(Value::OutputList(vec![x_value, y_value])),
        Some(_) => Err(graph_error(
            "treelayout",
            "treelayout: too many output arguments; maximum is 2",
        )),
        None => Ok(x_value),
    }
}
