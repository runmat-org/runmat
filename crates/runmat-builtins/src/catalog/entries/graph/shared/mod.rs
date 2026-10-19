//! Shared graph class identity, descriptor schema, and reusable inference facts.

use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_types::StaticClassIdentity;

pub const GRAPH_CLASS: StaticClassIdentity = StaticClassIdentity::new("graph");
pub const DIGRAPH_CLASS: StaticClassIdentity = StaticClassIdentity::new("digraph");

pub const GRAPH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAPH.INVALID_ARGUMENT",
    identifier: Some("RunMat:graph:InvalidArgument"),
    when: "Graph arguments, node ids, node names, edge weights, or output counts are invalid.",
    message: "graph: invalid argument",
};
pub const GRAPH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAPH.INTERNAL",
    identifier: Some("RunMat:graph:Internal"),
    when: "RunMat cannot build a requested graph value or query output.",
    message: "graph: internal error",
};
pub const GRAPH_ERRORS: [BuiltinErrorDescriptor; 2] =
    [GRAPH_ERROR_INVALID_ARGUMENT, GRAPH_ERROR_INTERNAL];

mod facts;
pub(crate) use facts::{
    adjacency_argument, arity, constructor_arguments, double_fact, graph_object_fact,
    graph_operand, node_list_fact, node_list_selector, node_selector, parent_vector,
};
mod integer;
pub use integer::{
    DIGRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES, DIGRAPH_INTEGER_ADJACENCY_EXTENSION,
    DIGRAPH_INTEGER_EXTENSIONS, DIGRAPH_INTEGER_WEIGHTS_EXTENSION,
    GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES, GRAPH_ENDPOINT_INTEGER_CAPABILITIES,
    GRAPH_INTEGER_ADJACENCY_EXTENSION, GRAPH_INTEGER_EXTENSIONS, GRAPH_INTEGER_WEIGHTS_EXTENSION,
    GRAPH_NODE_INTEGER_CAPABILITIES, GRAPH_NODE_LIST_INTEGER_CAPABILITIES,
    GRAPH_START_INTEGER_CAPABILITIES, TREE_LAYOUT_INTEGER_CAPABILITIES,
};

pub const fn param(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

pub const fn signature(
    label: &'static str,
    inputs: &'static [BuiltinParamDescriptor],
    outputs: &'static [BuiltinParamDescriptor],
) -> BuiltinSignatureDescriptor {
    BuiltinSignatureDescriptor {
        label,
        inputs,
        outputs,
    }
}

pub const fn descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
    output_mode: BuiltinOutputMode,
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &GRAPH_ERRORS,
    }
}
