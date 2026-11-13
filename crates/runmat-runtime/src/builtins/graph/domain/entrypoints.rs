use super::admission::{
    admit_distance_selectors, admit_scalar_selectors, constructor_arguments, count_as_double,
    graph_data_async, graph_edge_count, graph_node_count, resolve_distance_selectors,
    resolve_scalar_selectors,
};
use super::algorithms::{
    adjacency_lists, adjacency_tensor, degree_counts, dijkstra, edge_matches, indegree_counts,
    neighbor_query, neighbors_for, node_list_value, node_metric_value, outdegree_counts,
    reconstruct_path, tensor_value, zeroed_f64, NeighborMode,
};
use super::construction::construct_graph;
use super::parsing::numeric_vector;
use super::tree_layout::tree_coordinates;
use super::*;

#[derive(Clone, Copy)]
enum TwoOutputRequest {
    Discarded,
    Single,
    Pair,
    Default,
}

fn checked_two_output_request(name: &'static str) -> BuiltinResult<TwoOutputRequest> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(TwoOutputRequest::Discarded),
        Some(1) => Ok(TwoOutputRequest::Single),
        Some(2) => Ok(TwoOutputRequest::Pair),
        None => Ok(TwoOutputRequest::Default),
        Some(_) => Err(graph_error(
            name,
            format!("{name}: too many output arguments; maximum is 2"),
        )),
    }
}

#[runtime_builtin(
    name = "graph",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph",
    context = true
)]
pub(super) async fn graph_builtin_context(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = constructor_arguments(args, runtime.as_ref(), false, GRAPH_NAME).await?;
    construct_graph(GRAPH_NAME, false, args.view())
}

#[runtime_builtin(
    name = "digraph",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph",
    context = true
)]
pub(super) async fn digraph_builtin_context(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = constructor_arguments(args, runtime.as_ref(), true, DIGRAPH_NAME).await?;
    construct_graph(DIGRAPH_NAME, true, args.view())
}

#[cfg(test)]
pub(super) async fn graph_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    graph_builtin_context(None, args).await
}

#[cfg(test)]
pub(super) async fn digraph_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    digraph_builtin_context(None, args).await
}

#[runtime_builtin(
    name = "numnodes",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn numnodes_builtin(graph: Value) -> BuiltinResult<Value> {
    Ok(Value::Num(count_as_double(
        graph_node_count(&graph, "numnodes").await?,
        "numnodes",
    )?))
}

#[runtime_builtin(
    name = "numedges",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn numedges_builtin(graph: Value) -> BuiltinResult<Value> {
    Ok(Value::Num(count_as_double(
        graph_edge_count(&graph, "numedges")?,
        "numedges",
    )?))
}

#[runtime_builtin(
    name = "adjacency",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn adjacency_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data_async(&graph, "adjacency").await?;
    Ok(Value::Tensor(adjacency_tensor(&graph, "adjacency")?))
}

#[runtime_builtin(
    name = "degree",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn degree_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data_async(&graph, "degree").await?;
    node_metric_value(degree_counts(&graph), "degree")
}

#[runtime_builtin(
    name = "indegree",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn indegree_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data_async(&graph, "indegree").await?;
    node_metric_value(indegree_counts(&graph), "indegree")
}

#[runtime_builtin(
    name = "outdegree",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn outdegree_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data_async(&graph, "outdegree").await?;
    node_metric_value(outdegree_counts(&graph), "outdegree")
}

#[runtime_builtin(
    name = "neighbors",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn neighbors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
    neighbor_query(graph, node, NeighborMode::Undirected, "neighbors").await
}

#[runtime_builtin(
    name = "successors",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn successors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
    neighbor_query(graph, node, NeighborMode::Successors, "successors").await
}

#[runtime_builtin(
    name = "predecessors",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn predecessors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
    neighbor_query(graph, node, NeighborMode::Predecessors, "predecessors").await
}

#[runtime_builtin(
    name = "findedge",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn findedge_builtin(
    graph: Value,
    source: Value,
    target: Value,
) -> BuiltinResult<Value> {
    let selectors = admit_scalar_selectors(vec![source, target], "findedge")?;
    let graph = graph_data_async(&graph, "findedge").await?;
    let nodes = resolve_scalar_selectors(&graph, selectors, "findedge").await?;
    let source = nodes[0];
    let target = nodes[1];
    for (idx, edge) in graph.edges.iter().enumerate() {
        if edge_matches(edge, source, target, graph.directed) {
            return Ok(Value::Num((idx + 1) as f64));
        }
    }
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "shortestpath",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn shortestpath_builtin(
    graph: Value,
    source: Value,
    target: Value,
) -> BuiltinResult<Value> {
    let output_count = checked_two_output_request("shortestpath")?;
    let selectors = admit_scalar_selectors(vec![source, target], "shortestpath")?;
    let graph = graph_data_async(&graph, "shortestpath").await?;
    let nodes = resolve_scalar_selectors(&graph, selectors, "shortestpath").await?;
    let source = nodes[0];
    let target = nodes[1];
    let (distances, previous) = dijkstra(&graph, source);
    let path = reconstruct_path(source, target, &previous);
    let path_value = node_list_value(&graph, &path, "shortestpath")?;
    let distance = distances[target];
    match output_count {
        TwoOutputRequest::Discarded => Ok(Value::output_list(Vec::new())),
        TwoOutputRequest::Single => Ok(Value::output_list(vec![path_value])),
        TwoOutputRequest::Pair => Ok(Value::output_list(vec![path_value, Value::Num(distance)])),
        TwoOutputRequest::Default => Ok(path_value),
    }
}

#[runtime_builtin(
    name = "distances",
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn distances_builtin(
    graph: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let selectors = admit_distance_selectors(rest)?;
    let graph = graph_data_async(&graph, "distances").await?;
    let (sources, targets) = resolve_distance_selectors(&graph, selectors).await?;
    let cells = sources
        .len()
        .checked_mul(targets.len())
        .ok_or_else(|| graph_error("distances", "distances: result dimensions overflow"))?;
    let mut data = zeroed_f64(cells, "distances")?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn conncomp_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data_async(&graph, "conncomp").await?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn toposort_builtin(graph: Value) -> BuiltinResult<Value> {
    let graph = graph_data_async(&graph, "toposort").await?;
    if !graph.directed {
        return Err(graph_error(
            "toposort",
            "toposort: expected a digraph input",
        ));
    }
    let mut indegree = vec![0usize; graph.node_count];
    let mut adj = vec![Vec::new(); graph.node_count];
    for edge in &graph.edges {
        indegree[edge.target] += 1;
        adj[edge.source].push(edge.target);
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
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn bfsearch_builtin(graph: Value, start: Value) -> BuiltinResult<Value> {
    let selectors = admit_scalar_selectors(vec![start], "bfsearch")?;
    let graph = graph_data_async(&graph, "bfsearch").await?;
    let start = resolve_scalar_selectors(&graph, selectors, "bfsearch").await?[0];
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
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn dfsearch_builtin(graph: Value, start: Value) -> BuiltinResult<Value> {
    let selectors = admit_scalar_selectors(vec![start], "dfsearch")?;
    let graph = graph_data_async(&graph, "dfsearch").await?;
    let start = resolve_scalar_selectors(&graph, selectors, "dfsearch").await?[0];
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
    binding_variant = "default",
    builtin_path = "crate::builtins::graph"
)]
pub(super) async fn treelayout_builtin(parents: Value) -> BuiltinResult<Value> {
    let output_count = checked_two_output_request("treelayout")?;
    let parents = gather_if_needed_async(&parents).await?;
    let parents = numeric_vector(&parents, "treelayout")?;
    let n = parents.len();
    let (x, y) = tree_coordinates(&parents)?;
    let x_value = tensor_value(x, vec![1, n], "treelayout")?;
    let y_value = tensor_value(y, vec![1, n], "treelayout")?;
    match output_count {
        TwoOutputRequest::Discarded => Ok(Value::output_list(Vec::new())),
        TwoOutputRequest::Single => Ok(Value::output_list(vec![x_value])),
        TwoOutputRequest::Pair => Ok(Value::output_list(vec![x_value, y_value])),
        TwoOutputRequest::Default => Ok(x_value),
    }
}
