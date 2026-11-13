use runmat_types::MemberAccess;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

use runmat_builtins::{
    BuiltinErrorDescriptor, DIGRAPH_CLASS, GRAPH_CLASS, GRAPH_ERROR_INTERNAL,
    GRAPH_ERROR_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, ObjectInstance, StringArray, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NUM_NODES_PROPERTY: &str = "NumNodes";
const GRAPH_NAME: &str = "graph";
const DIGRAPH_NAME: &str = "digraph";

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

fn graph_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    error_from_descriptor(name, GRAPH_ERROR_INVALID_ARGUMENT, message)
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    error_from_descriptor(name, GRAPH_ERROR_INTERNAL, message)
}

fn error_from_descriptor(
    name: &'static str,
    descriptor: BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let builder = build_runtime_error(message).with_builtin(name);
    match descriptor.identifier {
        Some(identifier) => builder.with_identifier(identifier).build(),
        None => builder.build(),
    }
}

mod algorithms;
mod admission;
mod class_registry;
mod construction;
mod entrypoints;
mod parsing;
mod tree_layout;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use entrypoints::{
    __runmat_wasm_register_builtin_adjacency_builtin,
    __runmat_wasm_register_builtin_bfsearch_builtin,
    __runmat_wasm_register_builtin_conncomp_builtin, __runmat_wasm_register_builtin_degree_builtin,
    __runmat_wasm_register_builtin_dfsearch_builtin,
    __runmat_wasm_register_builtin_digraph_builtin_context,
    __runmat_wasm_register_builtin_distances_builtin,
    __runmat_wasm_register_builtin_findedge_builtin,
    __runmat_wasm_register_builtin_graph_builtin_context,
    __runmat_wasm_register_builtin_indegree_builtin,
    __runmat_wasm_register_builtin_neighbors_builtin,
    __runmat_wasm_register_builtin_numedges_builtin,
    __runmat_wasm_register_builtin_numnodes_builtin,
    __runmat_wasm_register_builtin_outdegree_builtin,
    __runmat_wasm_register_builtin_predecessors_builtin,
    __runmat_wasm_register_builtin_shortestpath_builtin,
    __runmat_wasm_register_builtin_successors_builtin,
    __runmat_wasm_register_builtin_toposort_builtin,
    __runmat_wasm_register_builtin_treelayout_builtin,
};
