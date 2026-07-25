//! MATLAB-compatible graph and digraph data-type helpers.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::OnceLock;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, IntValue, MethodDef, ObjectInstance, PropertyDef, ResolveContext, StringArray,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::table::{table_from_columns, table_variables};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const GRAPH_CLASS: &str = "graph";
const DIGRAPH_CLASS: &str = "digraph";
const NUM_NODES_PROPERTY: &str = "NumNodes";
const GRAPH_NAME: &str = "graph";
const DIGRAPH_NAME: &str = "digraph";
const INTERNAL_ERROR_ID: &str = "RunMat:graph:Internal";
const INVALID_ERROR_ID: &str = "RunMat:graph:InvalidArgument";

const INPUT_GRAPH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graph or digraph object.",
};
const INPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
};
const INPUT_REST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Additional arguments.",
};
const OUTPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
};
const OUTPUTS_ONE: [BuiltinParamDescriptor; 1] = [OUTPUT_VALUE];
const INPUTS_CONSTRUCTOR: [BuiltinParamDescriptor; 2] = [INPUT_VALUE, INPUT_REST];
const INPUTS_GRAPH: [BuiltinParamDescriptor; 1] = [INPUT_GRAPH];
const INPUTS_GRAPH_REST: [BuiltinParamDescriptor; 2] = [INPUT_GRAPH, INPUT_REST];

const GRAPH_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "G = graph(s, t)",
        inputs: &INPUTS_CONSTRUCTOR,
        outputs: &OUTPUTS_ONE,
    },
    BuiltinSignatureDescriptor {
        label: "G = graph(s, t, weights, names)",
        inputs: &INPUTS_CONSTRUCTOR,
        outputs: &OUTPUTS_ONE,
    },
    BuiltinSignatureDescriptor {
        label: "G = graph(A)",
        inputs: &INPUTS_CONSTRUCTOR,
        outputs: &OUTPUTS_ONE,
    },
];
const DIGRAPH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "G = digraph(s, t)",
        inputs: &INPUTS_CONSTRUCTOR,
        outputs: &OUTPUTS_ONE,
    },
    BuiltinSignatureDescriptor {
        label: "G = digraph(A)",
        inputs: &INPUTS_CONSTRUCTOR,
        outputs: &OUTPUTS_ONE,
    },
];
const UNARY_GRAPH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = f(G)",
    inputs: &INPUTS_GRAPH,
    outputs: &OUTPUTS_ONE,
}];
const GRAPH_QUERY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = f(G, args)",
    inputs: &INPUTS_GRAPH_REST,
    outputs: &OUTPUTS_ONE,
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAPH.INVALID_ARGUMENT",
    identifier: Some(INVALID_ERROR_ID),
    when: "Graph arguments, node ids, node names, edge weights, or output counts are invalid.",
    message: "graph: invalid argument",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAPH.INTERNAL",
    identifier: Some(INTERNAL_ERROR_ID),
    when: "RunMat cannot build the requested graph value or query output.",
    message: "graph: internal error",
};
const GRAPH_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const GRAPH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRAPH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAPH_ERRORS,
};
pub const DIGRAPH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIGRAPH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAPH_ERRORS,
};
pub const GRAPH_UNARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNARY_GRAPH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAPH_ERRORS,
};
pub const GRAPH_QUERY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRAPH_QUERY_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAPH_ERRORS,
};

#[derive(Clone, Debug)]
struct GraphData {
    directed: bool,
    node_names: Option<Vec<String>>,
    edges: Vec<Edge>,
    node_count: usize,
}

#[derive(Clone, Debug)]
struct Edge {
    source: usize,
    target: usize,
    weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct QueueItem {
    distance: f64,
    node: usize,
}

impl Eq for QueueItem {}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn numeric_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

fn graph_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(INVALID_ERROR_ID)
        .build()
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(INTERNAL_ERROR_ID)
        .build()
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(values.len());
    for value in values {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

#[runtime_builtin(
    name = "graph",
    category = "graph",
    summary = "Create an undirected graph object.",
    keywords = "graph,edges,nodes,network",
    type_resolver(any_type),
    descriptor(crate::builtins::graph::GRAPH_DESCRIPTOR),
    builtin_path = "crate::builtins::graph"
)]
async fn graph_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
async fn digraph_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
async fn numnodes_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn numedges_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn adjacency_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn degree_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn indegree_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn outdegree_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn neighbors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
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
async fn successors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
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
async fn predecessors_builtin(graph: Value, node: Value) -> BuiltinResult<Value> {
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
async fn findedge_builtin(graph: Value, source: Value, target: Value) -> BuiltinResult<Value> {
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
async fn shortestpath_builtin(graph: Value, source: Value, target: Value) -> BuiltinResult<Value> {
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
async fn distances_builtin(graph: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
async fn conncomp_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn toposort_builtin(graph: Value) -> BuiltinResult<Value> {
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
async fn bfsearch_builtin(graph: Value, start: Value) -> BuiltinResult<Value> {
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
async fn dfsearch_builtin(graph: Value, start: Value) -> BuiltinResult<Value> {
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
async fn treelayout_builtin(parents: Value) -> BuiltinResult<Value> {
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

fn construct_graph(name: &'static str, directed: bool, args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return graph_object(GraphData {
            directed,
            node_names: None,
            edges: Vec::new(),
            node_count: 0,
        });
    }
    if args.len() == 1 || is_adjacency_form(&args) {
        return graph_from_adjacency(name, directed, &args);
    }
    graph_from_edges(name, directed, &args)
}

fn is_adjacency_form(args: &[Value]) -> bool {
    matches!(args.first(), Some(Value::Tensor(t)) if t.rows == t.cols)
        && (args.len() == 1
            || args
                .get(1)
                .is_some_and(|arg| parse_string_vector(arg).is_ok() || scalar_text(arg).is_some()))
}

fn graph_from_adjacency(
    name: &'static str,
    directed: bool,
    args: &[Value],
) -> BuiltinResult<Value> {
    let Value::Tensor(matrix) = &args[0] else {
        return Err(graph_error(
            name,
            format!("{name}: expected adjacency matrix"),
        ));
    };
    if matrix.rows != matrix.cols {
        return Err(graph_error(
            name,
            format!("{name}: adjacency matrix must be square"),
        ));
    }
    let node_count = matrix.rows;
    let mut mode = TriangleMode::Full;
    let mut node_names = None;
    if let Some(arg) = args.get(1) {
        if let Some(text) = scalar_text(arg) {
            match canonical(&text).as_str() {
                "upper" => mode = TriangleMode::Upper,
                "lower" => mode = TriangleMode::Lower,
                _ => node_names = Some(parse_node_names(arg, node_count, name)?),
            }
        } else {
            node_names = Some(parse_node_names(arg, node_count, name)?);
        }
    }
    let mut edges = Vec::new();
    for col in 0..node_count {
        for row in 0..node_count {
            let weight = if directed {
                matrix.data[row + col * matrix.rows]
            } else {
                match mode {
                    TriangleMode::Full => {
                        if row > col {
                            continue;
                        }
                        let upper = matrix.data[row + col * matrix.rows];
                        if upper != 0.0 {
                            upper
                        } else if row == col {
                            0.0
                        } else {
                            matrix.data[col + row * matrix.rows]
                        }
                    }
                    TriangleMode::Upper => {
                        if row > col {
                            continue;
                        }
                        matrix.data[row + col * matrix.rows]
                    }
                    TriangleMode::Lower => {
                        if row < col {
                            continue;
                        }
                        matrix.data[row + col * matrix.rows]
                    }
                }
            };
            if weight != 0.0 {
                edges.push(Edge {
                    source: row,
                    target: col,
                    weight,
                });
            }
        }
    }
    graph_object(GraphData {
        directed,
        node_names,
        edges,
        node_count,
    })
}

fn graph_from_edges(name: &'static str, directed: bool, args: &[Value]) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(graph_error(
            name,
            format!("{name}: expected source and target node lists"),
        ));
    }
    let source_nodes = node_tokens(&args[0], name)?;
    let target_nodes = node_tokens(&args[1], name)?;
    if source_nodes.len() != target_nodes.len() {
        return Err(graph_error(
            name,
            format!("{name}: source and target lists must have the same length"),
        ));
    }
    let weights = match args.get(2) {
        Some(value) if looks_like_weights(value, source_nodes.len()) => {
            weights_from_value(value, source_nodes.len(), name)?
        }
        _ => vec![1.0; source_nodes.len()],
    };
    let names_arg = if args.len() >= 4 {
        args.get(3)
    } else if args.len() == 3 && !looks_like_weights(&args[2], source_nodes.len()) {
        args.get(2)
    } else {
        None
    };

    let mut node_names = names_arg
        .map(|value| parse_string_vector(value))
        .transpose()
        .map_err(|err| graph_error(name, format!("{name}: {err}")))?;
    let mut node_count = node_names.as_ref().map_or(0, Vec::len);
    let mut label_to_index = HashMap::<String, usize>::new();
    if let Some(names) = &node_names {
        for (idx, node_name) in names.iter().enumerate() {
            label_to_index.insert(node_name.clone(), idx);
        }
    }

    let mut edges = Vec::with_capacity(source_nodes.len());
    for ((source, target), weight) in source_nodes.into_iter().zip(target_nodes).zip(weights) {
        let source = resolve_node_token(
            source,
            &mut node_names,
            &mut label_to_index,
            &mut node_count,
            name,
        )?;
        let target = resolve_node_token(
            target,
            &mut node_names,
            &mut label_to_index,
            &mut node_count,
            name,
        )?;
        edges.push(Edge {
            source,
            target,
            weight,
        });
    }

    graph_object(GraphData {
        directed,
        node_names,
        edges,
        node_count,
    })
}

fn graph_object(graph: GraphData) -> BuiltinResult<Value> {
    ensure_graph_classes_registered();
    let class_name = if graph.directed {
        DIGRAPH_CLASS
    } else {
        GRAPH_CLASS
    };
    let mut object = ObjectInstance::new(class_name.to_string());
    object.properties.insert(
        NUM_NODES_PROPERTY.to_string(),
        Value::Num(graph.node_count as f64),
    );
    object
        .properties
        .insert("Edges".to_string(), edges_table(&graph)?);
    object
        .properties
        .insert("Nodes".to_string(), nodes_table(&graph)?);
    Ok(Value::Object(object))
}

fn edges_table(graph: &GraphData) -> BuiltinResult<Value> {
    let names = graph.node_names.as_ref();
    let end_nodes = if let Some(node_names) = names {
        let mut values = Vec::with_capacity(graph.edges.len() * 2);
        values.extend(
            graph
                .edges
                .iter()
                .map(|edge| node_names[edge.source].clone()),
        );
        values.extend(
            graph
                .edges
                .iter()
                .map(|edge| node_names[edge.target].clone()),
        );
        Value::StringArray(
            StringArray::new(values, vec![graph.edges.len(), 2])
                .map_err(|err| internal_error("graph", err))?,
        )
    } else {
        let mut values = Vec::with_capacity(graph.edges.len() * 2);
        values.extend(graph.edges.iter().map(|edge| (edge.source + 1) as f64));
        values.extend(graph.edges.iter().map(|edge| (edge.target + 1) as f64));
        tensor_value_raw(values, vec![graph.edges.len(), 2], "graph")?
    };
    let all_unit = graph.edges.iter().all(|edge| edge.weight == 1.0);
    if all_unit {
        table_from_columns(vec!["EndNodes".to_string()], vec![end_nodes])
    } else {
        let weights: Vec<f64> = graph.edges.iter().map(|edge| edge.weight).collect();
        table_from_columns(
            vec!["EndNodes".to_string(), "Weight".to_string()],
            vec![
                end_nodes,
                tensor_value_raw(weights, vec![graph.edges.len(), 1], "graph")?,
            ],
        )
    }
}

fn nodes_table(graph: &GraphData) -> BuiltinResult<Value> {
    if let Some(names) = &graph.node_names {
        table_from_columns(
            vec!["Name".to_string()],
            vec![Value::StringArray(
                StringArray::new(names.clone(), vec![graph.node_count, 1])
                    .map_err(|err| internal_error("graph", err))?,
            )],
        )
    } else {
        table_from_columns(
            vec!["Index".to_string()],
            vec![tensor_value_raw(
                (1..=graph.node_count).map(|idx| idx as f64).collect(),
                vec![graph.node_count, 1],
                "graph",
            )?],
        )
    }
}

fn graph_data(value: &Value, name: &'static str) -> BuiltinResult<GraphData> {
    let object = match value {
        Value::Object(object)
            if object.class_name == GRAPH_CLASS || object.class_name == DIGRAPH_CLASS =>
        {
            object
        }
        other => {
            return Err(graph_error(
                name,
                format!("{name}: expected graph or digraph object, got {other:?}"),
            ))
        }
    };
    let directed = object.class_name == DIGRAPH_CLASS;
    let node_count = match object.properties.get(NUM_NODES_PROPERTY) {
        Some(value) => positive_integer_scalar(value, name)?,
        None => infer_node_count_from_nodes(object, name)?,
    };
    let node_names = node_names_from_object(object, node_count, name)?;
    let edges = edges_from_object(object, node_names.as_deref(), directed, node_count, name)?;
    Ok(GraphData {
        directed,
        node_names,
        edges,
        node_count,
    })
}

fn node_names_from_object(
    object: &ObjectInstance,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<Option<Vec<String>>> {
    let Some(Value::Object(nodes)) = object.properties.get("Nodes") else {
        return Ok(None);
    };
    let variables = table_variables(nodes)
        .map_err(|err| graph_error(name, format!("{name}: invalid Nodes table: {err}")))?;
    let Some(value) = variables.fields.get("Name") else {
        return Ok(None);
    };
    let names = parse_string_vector(value).map_err(|err| graph_error(name, err))?;
    if names.len() != node_count {
        return Err(graph_error(
            name,
            format!("{name}: Nodes.Name length does not match graph node count"),
        ));
    }
    Ok(Some(names))
}

fn infer_node_count_from_nodes(
    object: &ObjectInstance,
    name: &'static str,
) -> BuiltinResult<usize> {
    let Some(Value::Object(nodes)) = object.properties.get("Nodes") else {
        return Ok(0);
    };
    crate::builtins::table::table_height(nodes)
        .map_err(|err| graph_error(name, format!("{name}: invalid Nodes table: {err}")))
}

fn edges_from_object(
    object: &ObjectInstance,
    node_names: Option<&[String]>,
    directed: bool,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<Vec<Edge>> {
    let Some(Value::Object(edges_table)) = object.properties.get("Edges") else {
        return Ok(Vec::new());
    };
    let variables = table_variables(edges_table)
        .map_err(|err| graph_error(name, format!("{name}: invalid Edges table: {err}")))?;
    let Some(endnodes) = variables.fields.get("EndNodes") else {
        return Ok(Vec::new());
    };
    let endpoint_pairs = edge_pairs_from_value(endnodes, node_names, node_count, name)?;
    let weights = match variables.fields.get("Weight") {
        Some(value) => weights_from_value(value, endpoint_pairs.len(), name)?,
        None => vec![1.0; endpoint_pairs.len()],
    };
    let mut edges = Vec::with_capacity(endpoint_pairs.len());
    for ((source, target), weight) in endpoint_pairs.into_iter().zip(weights) {
        if source >= node_count || target >= node_count {
            return Err(graph_error(
                name,
                format!("{name}: edge endpoint exceeds graph node count"),
            ));
        }
        edges.push(Edge {
            source,
            target,
            weight,
        });
    }
    if !directed {
        for edge in &edges {
            if edge.source >= node_count || edge.target >= node_count {
                return Err(graph_error(name, format!("{name}: invalid edge endpoint")));
            }
        }
    }
    Ok(edges)
}

fn edge_pairs_from_value(
    value: &Value,
    node_names: Option<&[String]>,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<Vec<(usize, usize)>> {
    match value {
        Value::Tensor(tensor) if tensor.cols == 2 => {
            let mut out = Vec::with_capacity(tensor.rows);
            for row in 0..tensor.rows {
                let source = node_index_from_tensor(tensor, row, node_count, name)?;
                let target = node_index_from_tensor(tensor, row + tensor.rows, node_count, name)?;
                out.push((source, target));
            }
            Ok(out)
        }
        Value::StringArray(array) if array.cols == 2 => {
            let Some(names) = node_names else {
                return Err(graph_error(
                    name,
                    format!("{name}: string EndNodes require Nodes.Name values"),
                ));
            };
            let lookup: HashMap<&str, usize> = names
                .iter()
                .enumerate()
                .map(|(idx, text)| (text.as_str(), idx))
                .collect();
            let mut out = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let source = *lookup.get(array.data[row].as_str()).ok_or_else(|| {
                    graph_error(
                        name,
                        format!("{name}: unknown source node '{}'", array.data[row]),
                    )
                })?;
                let target = *lookup
                    .get(array.data[row + array.rows].as_str())
                    .ok_or_else(|| {
                        graph_error(
                            name,
                            format!(
                                "{name}: unknown target node '{}'",
                                array.data[row + array.rows]
                            ),
                        )
                    })?;
                out.push((source, target));
            }
            Ok(out)
        }
        other => Err(graph_error(
            name,
            format!(
                "{name}: Edges.EndNodes must be an m-by-2 numeric or string array, got {other:?}"
            ),
        )),
    }
}

fn node_tokens(value: &Value, name: &'static str) -> BuiltinResult<Vec<NodeToken>> {
    if let Ok(values) = numeric_vector(value, name) {
        return Ok(values.into_iter().map(NodeToken::Index).collect());
    }
    Ok(parse_string_vector(value)
        .map_err(|err| graph_error(name, format!("{name}: {err}")))?
        .into_iter()
        .map(NodeToken::Name)
        .collect())
}

#[derive(Debug)]
enum NodeToken {
    Index(usize),
    Name(String),
}

fn resolve_node_token(
    token: NodeToken,
    node_names: &mut Option<Vec<String>>,
    label_to_index: &mut HashMap<String, usize>,
    node_count: &mut usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    match token {
        NodeToken::Index(index) => {
            if index == 0 {
                return Err(graph_error(
                    name,
                    format!("{name}: node indices are one-based"),
                ));
            }
            let zero = index - 1;
            *node_count = (*node_count).max(index);
            if let Some(names) = node_names {
                if zero >= names.len() {
                    return Err(graph_error(
                        name,
                        format!("{name}: numeric node index exceeds supplied node names"),
                    ));
                }
            }
            Ok(zero)
        }
        NodeToken::Name(label) => {
            if node_names.is_none() {
                *node_names = Some(Vec::new());
            }
            if let Some(&idx) = label_to_index.get(&label) {
                return Ok(idx);
            }
            let idx = node_names.as_ref().map_or(0, Vec::len);
            label_to_index.insert(label.clone(), idx);
            node_names.as_mut().expect("node names").push(label);
            *node_count = (*node_count).max(idx + 1);
            Ok(idx)
        }
    }
}

fn looks_like_weights(value: &Value, expected_len: usize) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) => expected_len == 1,
        Value::Tensor(tensor) => tensor.data.len() == expected_len,
        _ => false,
    }
}

fn weights_from_value(value: &Value, len: usize, name: &'static str) -> BuiltinResult<Vec<f64>> {
    let values = numeric_f64_vector(value, name)?;
    if values.len() != len {
        return Err(graph_error(
            name,
            format!("{name}: weights must have one value per edge"),
        ));
    }
    for weight in &values {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(graph_error(
                name,
                format!("{name}: edge weights must be finite nonnegative values"),
            ));
        }
    }
    Ok(values)
}

fn numeric_vector(value: &Value, name: &'static str) -> BuiltinResult<Vec<usize>> {
    if let Some(values) = exact_integer_vector(value, name) {
        return values;
    }
    let values = numeric_f64_vector(value, name)?;
    values
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value < 0.0 || value.fract().abs() > 1e-9 {
                Err(graph_error(
                    name,
                    format!("{name}: node indices must be nonnegative integers"),
                ))
            } else {
                Ok(value as usize)
            }
        })
        .collect()
}

fn numeric_f64_vector(value: &Value, name: &'static str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => Ok(tensor::tensor_values_f64(tensor)),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect()),
        other => Err(graph_error(
            name,
            format!("{name}: expected numeric vector, got {other:?}"),
        )),
    }
}

fn exact_integer_vector(value: &Value, name: &'static str) -> Option<BuiltinResult<Vec<usize>>> {
    match value {
        Value::Int(value) => Some(parse_nonnegative_integer(value, name).map(|value| vec![value])),
        Value::Tensor(tensor) => tensor.integer_storage().map(|storage| {
            storage
                .exact_values()
                .into_iter()
                .map(|value| parse_nonnegative_integer(&value, name))
                .collect()
        }),
        _ => None,
    }
}

fn parse_nonnegative_integer(value: &IntValue, name: &'static str) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        graph_error(
            name,
            format!("{name}: node indices must be nonnegative integers"),
        )
    })
}

fn parse_node_names(
    value: &Value,
    expected_len: usize,
    name: &'static str,
) -> BuiltinResult<Vec<String>> {
    let names = parse_string_vector(value).map_err(|err| graph_error(name, err))?;
    if names.len() != expected_len {
        return Err(graph_error(
            name,
            format!("{name}: node name list length must match node count"),
        ));
    }
    Ok(names)
}

fn parse_string_vector(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(vec![chars.data.iter().collect()]),
        Value::CharArray(chars) => Ok((0..chars.rows)
            .map(|row| {
                (0..chars.cols)
                    .map(|col| chars.data[row + col * chars.rows])
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|entry| {
                scalar_text(entry).ok_or_else(|| {
                    "expected cell array of string or character node names".to_string()
                })
            })
            .collect(),
        other => Err(format!(
            "expected string, character, or cellstr vector, got {other:?}"
        )),
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn positive_integer_scalar(value: &Value, name: &'static str) -> BuiltinResult<usize> {
    let values = numeric_vector(value, name)?;
    if values.len() != 1 {
        return Err(graph_error(
            name,
            format!("{name}: expected scalar node count"),
        ));
    }
    Ok(values[0])
}

fn node_index_from_value(
    graph: &GraphData,
    value: &Value,
    name: &'static str,
) -> BuiltinResult<usize> {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            let labels = parse_string_vector(value).map_err(|err| graph_error(name, err))?;
            if labels.len() != 1 {
                return Err(graph_error(name, format!("{name}: expected one node")));
            }
            let Some(names) = &graph.node_names else {
                return Err(graph_error(
                    name,
                    format!("{name}: string node references require named graph nodes"),
                ));
            };
            names
                .iter()
                .position(|candidate| candidate == &labels[0])
                .ok_or_else(|| graph_error(name, format!("{name}: unknown node '{}'", labels[0])))
        }
        _ => {
            let idx = positive_integer_scalar(value, name)?;
            node_index_from_one_based(idx, graph.node_count, name)
        }
    }
}

fn node_indices_from_value(
    graph: &GraphData,
    value: &Value,
    name: &'static str,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            let labels = parse_string_vector(value).map_err(|err| graph_error(name, err))?;
            let Some(names) = &graph.node_names else {
                return Err(graph_error(
                    name,
                    format!("{name}: string node references require named graph nodes"),
                ));
            };
            labels
                .into_iter()
                .map(|label| {
                    names
                        .iter()
                        .position(|candidate| candidate == &label)
                        .ok_or_else(|| graph_error(name, format!("{name}: unknown node '{label}'")))
                })
                .collect()
        }
        _ => numeric_vector(value, name)?
            .into_iter()
            .map(|idx| node_index_from_one_based(idx, graph.node_count, name))
            .collect(),
    }
}

fn node_index_from_tensor(
    tensor: &Tensor,
    index: usize,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(index)
            .ok_or_else(|| graph_error(name, format!("{name}: endpoint index is out of bounds")))?;
        return node_index_from_integer(&value, node_count, name);
    }
    node_index_from_f64(tensor.data[index], node_count, name)
}

fn node_index_from_integer(
    value: &IntValue,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    let Some(value) = value.try_to_usize() else {
        return Err(graph_error(
            name,
            format!("{name}: edge endpoints must be positive integer node ids"),
        ));
    };
    node_index_from_one_based(value, node_count, name)
}

fn node_index_from_f64(value: f64, node_count: usize, name: &'static str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract().abs() > 1e-9 {
        return Err(graph_error(
            name,
            format!("{name}: edge endpoints must be positive integer node ids"),
        ));
    }
    node_index_from_one_based(value as usize, node_count, name)
}

fn node_index_from_one_based(
    value: usize,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    if value == 0 || value > node_count {
        return Err(graph_error(
            name,
            format!("{name}: node index is outside the graph"),
        ));
    }
    Ok(value - 1)
}

fn edge_matches(edge: &Edge, source: usize, target: usize, directed: bool) -> bool {
    (edge.source == source && edge.target == target)
        || (!directed && edge.source == target && edge.target == source)
}

fn adjacency_tensor(graph: &GraphData, name: &'static str) -> BuiltinResult<Tensor> {
    let n = graph.node_count;
    let mut data = vec![0.0; n * n];
    for edge in &graph.edges {
        data[edge.source + edge.target * n] += edge.weight;
        if !graph.directed && edge.source != edge.target {
            data[edge.target + edge.source * n] += edge.weight;
        }
    }
    Tensor::new(data, vec![n, n]).map_err(|err| internal_error(name, err))
}

fn degree_counts(graph: &GraphData) -> Vec<f64> {
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

fn indegree_counts(graph: &GraphData) -> Vec<f64> {
    if !graph.directed {
        return degree_counts_undirected(graph);
    }
    let mut counts = vec![0.0; graph.node_count];
    for edge in &graph.edges {
        counts[edge.target] += 1.0;
    }
    counts
}

fn outdegree_counts(graph: &GraphData) -> Vec<f64> {
    if !graph.directed {
        return degree_counts_undirected(graph);
    }
    let mut counts = vec![0.0; graph.node_count];
    for edge in &graph.edges {
        counts[edge.source] += 1.0;
    }
    counts
}

fn degree_counts_undirected(graph: &GraphData) -> Vec<f64> {
    let mut counts = vec![0.0; graph.node_count];
    for edge in &graph.edges {
        counts[edge.source] += 1.0;
        if edge.source != edge.target {
            counts[edge.target] += 1.0;
        }
    }
    counts
}

fn node_metric_value(values: Vec<f64>, name: &'static str) -> BuiltinResult<Value> {
    let len = values.len();
    tensor_value(values, vec![len, 1], name)
}

#[derive(Clone, Copy)]
enum NeighborMode {
    Undirected,
    Successors,
    Predecessors,
}

async fn neighbor_query(
    graph: Value,
    node: Value,
    mode: NeighborMode,
    name: &'static str,
) -> BuiltinResult<Value> {
    let graph = graph_data(&graph, name)?;
    let node = gather_if_needed_async(&node).await?;
    let node = node_index_from_value(&graph, &node, name)?;
    let mut neighbors = neighbors_for(&graph, node, mode);
    neighbors.sort_unstable();
    neighbors.dedup();
    node_list_value(&graph, &neighbors, name)
}

fn neighbors_for(graph: &GraphData, node: usize, mode: NeighborMode) -> Vec<usize> {
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

fn adjacency_lists(graph: &GraphData, mode: NeighborMode) -> Vec<Vec<usize>> {
    let mut lists = vec![Vec::new(); graph.node_count];
    for node in 0..graph.node_count {
        let mut neighbors = neighbors_for(graph, node, mode);
        neighbors.sort_unstable();
        neighbors.dedup();
        lists[node] = neighbors;
    }
    lists
}

fn weighted_adjacency(graph: &GraphData) -> Vec<Vec<(usize, f64)>> {
    let mut lists = vec![Vec::new(); graph.node_count];
    for edge in &graph.edges {
        lists[edge.source].push((edge.target, edge.weight));
        if !graph.directed && edge.source != edge.target {
            lists[edge.target].push((edge.source, edge.weight));
        }
    }
    lists
}

fn dijkstra(graph: &GraphData, source: usize) -> (Vec<f64>, Vec<Option<usize>>) {
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

fn reconstruct_path(source: usize, target: usize, previous: &[Option<usize>]) -> Vec<usize> {
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

fn node_list_value(graph: &GraphData, nodes: &[usize], name: &'static str) -> BuiltinResult<Value> {
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

fn assign_tree_positions(
    node: usize,
    depth: f64,
    children: &[Vec<usize>],
    leaf_cursor: &mut f64,
    x: &mut [f64],
    y: &mut [f64],
) {
    y[node] = -depth;
    if children[node].is_empty() {
        x[node] = *leaf_cursor;
        *leaf_cursor += 1.0;
        return;
    }
    for &child in &children[node] {
        assign_tree_positions(child, depth + 1.0, children, leaf_cursor, x, y);
    }
    let first = children[node].first().copied().unwrap();
    let last = children[node].last().copied().unwrap();
    x[node] = (x[first] + x[last]) / 2.0;
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>, name: &'static str) -> BuiltinResult<Value> {
    Ok(tensor::tensor_into_value(
        Tensor::new(data, shape).map_err(|err| internal_error(name, err))?,
    ))
}

fn tensor_value_raw(data: Vec<f64>, shape: Vec<usize>, name: &'static str) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(data, shape).map_err(|err| internal_error(name, err))?,
    ))
}

#[derive(Clone, Copy)]
enum TriangleMode {
    Full,
    Upper,
    Lower,
}

fn ensure_graph_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        register_graph_class(GRAPH_CLASS);
        register_graph_class(DIGRAPH_CLASS);
    });
}

fn register_graph_class(name: &str) {
    let mut properties = HashMap::new();
    for property_name in ["Edges", "Nodes", NUM_NODES_PROPERTY] {
        properties.insert(
            property_name.to_string(),
            PropertyDef {
                name: property_name.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
    }
    runmat_builtins::register_class(ClassDef {
        name: name.to_string(),
        parent: None,
        properties,
        methods: HashMap::<String, MethodDef>::new(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn numeric(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("tensor"))
    }

    fn int_numeric(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor"))
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(t) => t.data,
            Value::Num(n) => vec![n],
            other => panic!("expected numeric value, got {other:?}"),
        }
    }

    #[test]
    fn graph_constructor_builds_edges_nodes_and_adjacency() {
        let graph = block_on(graph_builtin(vec![
            numeric(vec![1.0, 2.0, 3.0], 3, 1),
            numeric(vec![2.0, 3.0, 1.0], 3, 1),
        ]))
        .expect("graph");
        assert_eq!(
            block_on(numnodes_builtin(graph.clone())).unwrap(),
            Value::Num(3.0)
        );
        assert_eq!(
            block_on(numedges_builtin(graph.clone())).unwrap(),
            Value::Num(3.0)
        );

        let adjacency = block_on(adjacency_builtin(graph)).expect("adjacency");
        assert_eq!(
            tensor_data(adjacency),
            vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn graph_constructor_accepts_typed_integer_node_vectors() {
        let graph = block_on(graph_builtin(vec![
            int_numeric(IntegerStorage::I16(vec![1, 2, 3]), 3, 1),
            int_numeric(IntegerStorage::U16(vec![2, 3, 1]), 3, 1),
            int_numeric(IntegerStorage::U8(vec![1, 2, 3]), 3, 1),
        ]))
        .expect("graph");
        assert_eq!(
            block_on(numnodes_builtin(graph.clone())).unwrap(),
            Value::Num(3.0)
        );
        assert_eq!(
            tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
            vec![0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 3.0, 2.0, 0.0]
        );
    }

    #[test]
    fn digraph_preserves_direction_for_neighbors_and_degree() {
        let graph = block_on(digraph_builtin(vec![
            numeric(vec![1.0, 1.0, 2.0], 3, 1),
            numeric(vec![2.0, 3.0, 3.0], 3, 1),
        ]))
        .expect("digraph");
        assert_eq!(
            tensor_data(block_on(outdegree_builtin(graph.clone())).unwrap()),
            vec![2.0, 1.0, 0.0]
        );
        assert_eq!(
            tensor_data(block_on(indegree_builtin(graph.clone())).unwrap()),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(
            tensor_data(block_on(successors_builtin(graph.clone(), Value::Num(1.0))).unwrap()),
            vec![2.0, 3.0]
        );
        assert_eq!(
            tensor_data(block_on(predecessors_builtin(graph, Value::Num(3.0))).unwrap()),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn named_graph_returns_named_paths() {
        let names = Value::StringArray(
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).expect("names"),
        );
        let graph = block_on(graph_builtin(vec![
            numeric(vec![1.0, 2.0], 2, 1),
            numeric(vec![2.0, 3.0], 2, 1),
            numeric(vec![2.0, 1.0], 2, 1),
            names,
        ]))
        .expect("graph");
        let path = block_on(shortestpath_builtin(
            graph,
            Value::CharArray(runmat_builtins::CharArray::new_row("a")),
            Value::CharArray(runmat_builtins::CharArray::new_row("c")),
        ))
        .expect("path");
        let Value::StringArray(path) = path else {
            panic!("expected named path");
        };
        assert_eq!(path.data, vec!["a", "b", "c"]);
    }

    #[test]
    fn adjacency_constructor_and_distances_use_weights() {
        let graph = block_on(digraph_builtin(vec![numeric(
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0],
            3,
            3,
        )]))
        .expect("digraph");
        let distances = block_on(distances_builtin(
            graph,
            vec![Value::Num(1.0), Value::Num(3.0)],
        ))
        .expect("distances");
        assert_eq!(distances, Value::Num(3.0));
    }

    #[test]
    fn edge_endnodes_table_accepts_typed_integer_matrix() {
        let endnodes = int_numeric(IntegerStorage::U16(vec![1, 2, 2, 3]), 2, 2);
        let edges =
            table_from_columns(vec!["EndNodes".to_string()], vec![endnodes]).expect("edges table");
        let nodes = table_from_columns(
            vec!["Index".to_string()],
            vec![numeric(vec![1.0, 2.0, 3.0], 3, 1)],
        )
        .expect("nodes table");
        let mut graph = ObjectInstance::new(GRAPH_CLASS.to_string());
        graph
            .properties
            .insert(NUM_NODES_PROPERTY.to_string(), Value::Num(3.0));
        graph.properties.insert("Edges".to_string(), edges);
        graph.properties.insert("Nodes".to_string(), nodes);
        let graph = Value::Object(graph);
        assert_eq!(
            tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn undirected_adjacency_constructor_accepts_lower_triangle() {
        let graph =
            block_on(graph_builtin(vec![numeric(vec![0.0, 5.0, 0.0, 0.0], 2, 2)])).expect("graph");
        assert_eq!(
            block_on(numedges_builtin(graph.clone())).unwrap(),
            Value::Num(1.0)
        );
        assert_eq!(
            tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
            vec![0.0, 5.0, 5.0, 0.0]
        );
    }

    #[test]
    fn treelayout_returns_single_or_pair_by_output_count() {
        let result = block_on(treelayout_builtin(numeric(vec![0.0, 1.0, 1.0, 2.0], 1, 4)))
            .expect("treelayout");
        assert_eq!(tensor_data(result).len(), 4);

        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(treelayout_builtin(numeric(vec![0.0, 1.0, 1.0, 2.0], 1, 4)))
            .expect("treelayout");
        let Value::OutputList(values) = result else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 2);
        assert_eq!(tensor_data(values[0].clone()).len(), 4);
        assert_eq!(tensor_data(values[1].clone()), vec![-0.0, -1.0, -1.0, -2.0]);
    }

    #[test]
    fn treelayout_accepts_typed_integer_parent_vector() {
        let result = block_on(treelayout_builtin(int_numeric(
            IntegerStorage::I16(vec![0, 1, 1, 2]),
            1,
            4,
        )))
        .expect("treelayout");
        assert_eq!(tensor_data(result).len(), 4);
    }

    #[test]
    fn graph_numeric_vector_reads_large_typed_integer_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let value = int_numeric(IntegerStorage::U64(vec![wide, wide - 1]), 1, 2);
        match usize::try_from(wide) {
            Ok(wide_usize) => {
                assert_eq!(
                    numeric_vector(&value, "graph").expect("numeric vector"),
                    vec![wide_usize, wide_usize - 1]
                );
            }
            Err(_) => {
                assert!(numeric_vector(&value, "graph").is_err());
            }
        }
    }
}
