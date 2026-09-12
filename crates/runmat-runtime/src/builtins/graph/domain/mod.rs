use runmat_types::MemberAccess;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::OnceLock;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, ObjectInstance, StringArray, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::table::{table_from_columns, table_variables};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const GRAPH_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("graph");
const DIGRAPH_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("digraph");
const NUM_NODES_PROPERTY: &str = "NumNodes";
const GRAPH_NAME: &str = "graph";
const DIGRAPH_NAME: &str = "digraph";

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
    identifier: Some("RunMat:graph:InvalidArgument"),
    when: "Graph arguments, node ids, node names, edge weights, or output counts are invalid.",
    message: "graph: invalid argument",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAPH.INTERNAL",
    identifier: Some("RunMat:graph:Internal"),
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
    error_from_descriptor(name, ERROR_INVALID_ARGUMENT, message)
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    error_from_descriptor(name, ERROR_INTERNAL, message)
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

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(values.len());
    for value in values {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

mod algorithms;
mod class_registry;
mod construction;
mod entrypoints;
mod parsing;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use entrypoints::{
    __runmat_wasm_register_builtin_adjacency_builtin,
    __runmat_wasm_register_builtin_bfsearch_builtin,
    __runmat_wasm_register_builtin_conncomp_builtin, __runmat_wasm_register_builtin_degree_builtin,
    __runmat_wasm_register_builtin_dfsearch_builtin,
    __runmat_wasm_register_builtin_digraph_builtin,
    __runmat_wasm_register_builtin_distances_builtin,
    __runmat_wasm_register_builtin_findedge_builtin, __runmat_wasm_register_builtin_graph_builtin,
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
