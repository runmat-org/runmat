use super::admission::{admit_scalar_selectors, graph_data_async, resolve_scalar_selectors};
use super::*;

pub(super) fn edge_matches(edge: &Edge, source: usize, target: usize, directed: bool) -> bool {
    (edge.source == source && edge.target == target)
        || (!directed && edge.source == target && edge.target == source)
}

pub(super) fn adjacency_tensor(graph: &GraphData, name: &'static str) -> BuiltinResult<Tensor> {
    let n = graph.node_count;
    let cells = n
        .checked_mul(n)
        .ok_or_else(|| graph_error(name, format!("{name}: adjacency dimensions overflow")))?;
    let mut data = zeroed_f64(cells, name)?;
    for edge in &graph.edges {
        data[edge.source + edge.target * n] += edge.weight;
        if !graph.directed && edge.source != edge.target {
            data[edge.target + edge.source * n] += edge.weight;
        }
    }
    Tensor::new(data, vec![n, n]).map_err(|err| internal_error(name, err))
}

pub(super) fn degree_counts(graph: &GraphData) -> Vec<f64> {
    if graph.directed {
        outdegree_counts(graph)
            .into_iter()
            .zip(indegree_counts(graph))
            .map(|(out, incoming)| out + incoming)
            .collect()
    } else {
        let mut counts = vec![0.0; graph.node_count];
        for edge in &graph.edges {
            if edge.source == edge.target {
                counts[edge.source] += 2.0;
            } else {
                counts[edge.source] += 1.0;
                counts[edge.target] += 1.0;
            }
        }
        counts
    }
}

pub(super) fn indegree_counts(graph: &GraphData) -> Vec<f64> {
    if !graph.directed {
        return degree_counts_undirected(graph);
    }
    let mut counts = vec![0.0; graph.node_count];
    for edge in &graph.edges {
        counts[edge.target] += 1.0;
    }
    counts
}

pub(super) fn outdegree_counts(graph: &GraphData) -> Vec<f64> {
    if !graph.directed {
        return degree_counts_undirected(graph);
    }
    let mut counts = vec![0.0; graph.node_count];
    for edge in &graph.edges {
        counts[edge.source] += 1.0;
    }
    counts
}

pub(super) fn degree_counts_undirected(graph: &GraphData) -> Vec<f64> {
    let mut counts = vec![0.0; graph.node_count];
    for edge in &graph.edges {
        if edge.source == edge.target {
            counts[edge.source] += 2.0;
        } else {
            counts[edge.source] += 1.0;
            counts[edge.target] += 1.0;
        }
    }
    counts
}

pub(super) fn node_metric_value(values: Vec<f64>, name: &'static str) -> BuiltinResult<Value> {
    let len = values.len();
    tensor_value(values, vec![len, 1], name)
}

#[derive(Clone, Copy)]
pub(super) enum NeighborMode {
    Undirected,
    Successors,
    Predecessors,
}

pub(super) async fn neighbor_query(
    graph: Value,
    node: Value,
    mode: NeighborMode,
    name: &'static str,
) -> BuiltinResult<Value> {
    let selectors = admit_scalar_selectors(vec![node], name)?;
    let graph = graph_data_async(&graph, name).await?;
    let node = resolve_scalar_selectors(&graph, selectors, name).await?[0];
    let mut neighbors = neighbors_for(&graph, node, mode);
    neighbors.sort_unstable();
    neighbors.dedup();
    node_list_value(&graph, &neighbors, name)
}

pub(super) fn neighbors_for(graph: &GraphData, node: usize, mode: NeighborMode) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in &graph.edges {
        match mode {
            NeighborMode::Undirected => {
                if edge.source == node {
                    out.push(edge.target);
                }
                if edge.target == node && edge.source != node {
                    out.push(edge.source);
                }
            }
            NeighborMode::Successors => {
                if edge.source == node {
                    out.push(edge.target);
                }
                if !graph.directed && edge.target == node && edge.source != node {
                    out.push(edge.source);
                }
            }
            NeighborMode::Predecessors => {
                if edge.target == node {
                    out.push(edge.source);
                }
                if !graph.directed && edge.source == node && edge.target != node {
                    out.push(edge.target);
                }
            }
        }
    }
    out
}

pub(super) fn adjacency_lists(graph: &GraphData, mode: NeighborMode) -> Vec<Vec<usize>> {
    let mut lists = vec![Vec::new(); graph.node_count];
    for node in 0..graph.node_count {
        let mut neighbors = neighbors_for(graph, node, mode);
        neighbors.sort_unstable();
        neighbors.dedup();
        lists[node] = neighbors;
    }
    lists
}

pub(super) fn weighted_adjacency(graph: &GraphData) -> Vec<Vec<(usize, f64)>> {
    let mut lists = vec![Vec::new(); graph.node_count];
    for edge in &graph.edges {
        lists[edge.source].push((edge.target, edge.weight));
        if !graph.directed && edge.source != edge.target {
            lists[edge.target].push((edge.source, edge.weight));
        }
    }
    lists
}

pub(super) fn dijkstra(graph: &GraphData, source: usize) -> (Vec<f64>, Vec<Option<usize>>) {
    let adjacency = weighted_adjacency(graph);
    let mut distances = vec![f64::INFINITY; graph.node_count];
    let mut previous = vec![None; graph.node_count];
    let mut heap = BinaryHeap::new();
    distances[source] = 0.0;
    heap.push(QueueItem {
        distance: 0.0,
        node: source,
    });
    while let Some(QueueItem { distance, node }) = heap.pop() {
        if distance > distances[node] {
            continue;
        }
        for &(next, weight) in &adjacency[node] {
            let candidate = distance + weight;
            if candidate < distances[next] {
                distances[next] = candidate;
                previous[next] = Some(node);
                heap.push(QueueItem {
                    distance: candidate,
                    node: next,
                });
            }
        }
    }
    (distances, previous)
}

pub(super) fn reconstruct_path(
    source: usize,
    target: usize,
    previous: &[Option<usize>],
) -> Vec<usize> {
    if source == target {
        return vec![source];
    }
    let mut path = Vec::new();
    let mut current = target;
    path.push(current);
    while let Some(parent) = previous[current] {
        current = parent;
        path.push(current);
        if current == source {
            path.reverse();
            return path;
        }
    }
    Vec::new()
}

pub(super) fn node_list_value(
    graph: &GraphData,
    nodes: &[usize],
    name: &'static str,
) -> BuiltinResult<Value> {
    if let Some(names) = &graph.node_names {
        Ok(Value::StringArray(
            StringArray::new(
                nodes.iter().map(|&idx| names[idx].clone()).collect(),
                vec![nodes.len(), 1],
            )
            .map_err(|err| internal_error(name, err))?,
        ))
    } else {
        tensor_value(
            nodes.iter().map(|&idx| (idx + 1) as f64).collect(),
            vec![nodes.len(), 1],
            name,
        )
    }
}

pub(super) fn tensor_value(
    data: Vec<f64>,
    shape: Vec<usize>,
    name: &'static str,
) -> BuiltinResult<Value> {
    Ok(tensor::tensor_into_value(
        Tensor::new(data, shape).map_err(|err| internal_error(name, err))?,
    ))
}

pub(super) fn tensor_value_raw(
    data: Vec<f64>,
    shape: Vec<usize>,
    name: &'static str,
) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(data, shape).map_err(|err| internal_error(name, err))?,
    ))
}

pub(super) fn zeroed_f64(len: usize, name: &'static str) -> BuiltinResult<Vec<f64>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|error| graph_error(name, format!("{name}: result allocation failed: {error}")))?;
    data.resize(len, 0.0);
    Ok(data)
}
