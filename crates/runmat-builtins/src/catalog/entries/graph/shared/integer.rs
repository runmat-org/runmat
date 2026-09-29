//! Typed integer forms and compatibility gates for graph construction and queries.

use crate::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    ALL_INTEGER_CLASSES,
};

pub const GRAPH_INTEGER_ADJACENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "graph-integer-adjacency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "graph with a typed-integer adjacency matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GraphIntegerAdjacencyExtension"),
    };
pub const GRAPH_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "graph-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "graph with typed-integer edge weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GraphIntegerWeightsExtension"),
    };
pub const DIGRAPH_INTEGER_ADJACENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "digraph-integer-adjacency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "digraph with a typed-integer adjacency matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DigraphIntegerAdjacencyExtension"),
    };
pub const DIGRAPH_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "digraph-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "digraph with typed-integer edge weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DigraphIntegerWeightsExtension"),
    };
pub const GRAPH_INTEGER_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    GRAPH_INTEGER_ADJACENCY_EXTENSION,
    GRAPH_INTEGER_WEIGHTS_EXTENSION,
];
pub const DIGRAPH_INTEGER_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DIGRAPH_INTEGER_ADJACENCY_EXTENSION,
    DIGRAPH_INTEGER_WEIGHTS_EXTENSION,
];

const INTEGER_ADJACENCY_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "MATLAB documents single, double, and logical adjacency arrays; RunMat admits typed integers only when every stored weight is exactly representable as double.",
}];
const INTEGER_WEIGHT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "weights",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "MATLAB documents single and double edge weights; RunMat checks exact binary64 representability before storing a weighted graph.",
}];
const INTEGER_NODE_PAIR_INPUTS: [BuiltinIntegerInputCapability; 2] =
    [structural_integer_input("s"), structural_integer_input("t")];
pub const GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = graph(integer_A, ...)", inputs: &INTEGER_ADJACENCY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed adjacency is gated as a RunMat extension before provider access. The graph stores checked double edge weights.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = graph(s, t, integer_weights, ...)", inputs: &INTEGER_WEIGHT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed weights are gated as a RunMat extension before provider access, then checked at the double storage boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = graph(integer_s, integer_t, ...)", inputs: &INTEGER_NODE_PAIR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Numeric node indices are publicly documented without a per-class list. RunMat decodes all eight typed classes exactly and range-checks against host node capacity; MATLAB per-class parity remains unverified.",
    },
];

pub const DIGRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = digraph(integer_A, ...)", inputs: &INTEGER_ADJACENCY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed adjacency is gated as a RunMat extension before provider access. The graph stores checked double edge weights.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = digraph(s, t, integer_weights, ...)", inputs: &INTEGER_WEIGHT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed weights are gated as a RunMat extension before provider access, then checked at the double storage boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = digraph(integer_s, integer_t, ...)", inputs: &INTEGER_NODE_PAIR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Numeric node indices are publicly documented without a per-class list. RunMat decodes all eight typed classes exactly and range-checks against host node capacity; MATLAB per-class parity remains unverified.",
    },
];

const fn structural_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name, classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric node indices are publicly documented, but the fixed-width class matrix is not enumerated. RunMat's exact structural decoding is verified independently; per-class MATLAB parity is unverified.",
    }
}

const INTEGER_NODE_INPUT: [BuiltinIntegerInputCapability; 1] = [structural_integer_input("node")];
const INTEGER_START_INPUT: [BuiltinIntegerInputCapability; 1] = [structural_integer_input("start")];
const INTEGER_ENDPOINT_INPUTS: [BuiltinIntegerInputCapability; 2] =
    [structural_integer_input("s"), structural_integer_input("t")];
pub const GRAPH_NODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "graph_query(G, integer_node)", inputs: &INTEGER_NODE_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "The node selector is decoded exactly and range-checked without a double detour. Public docs describe numeric node indices but do not enumerate fixed-width classes; per-class MATLAB parity is unverified.",
}];
pub const GRAPH_START_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "graph_search(G, integer_start)", inputs: &INTEGER_START_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "The start node is decoded exactly and range-checked without a double detour. Public docs describe numeric node indices without a fixed-width class matrix; per-class MATLAB parity is unverified.",
}];
pub const GRAPH_ENDPOINT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "graph_query(G, integer_s, integer_t)", inputs: &INTEGER_ENDPOINT_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Both node selectors retain exact fixed-width values until host index validation; per-class MATLAB parity is unverified.",
}];
pub const GRAPH_NODE_LIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "D = distances(G, integer_sources, integer_targets)", inputs: &INTEGER_ENDPOINT_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Source and target lists are exact structural indices. Public docs describe numeric node IDs without a fixed-width class matrix; per-class MATLAB parity is unverified.",
}];
const INTEGER_PARENT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [structural_integer_input("parent")];
pub const TREE_LAYOUT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "[x, y] = treelayout(integer_parent)", inputs: &INTEGER_PARENT_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Parent indices are decoded exactly and checked for range and cycles. MATLAB documents a parent vector without enumerating fixed-width classes; per-class parity and layout coordinates are unverified.",
}];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_extensions_remain_distinct_from_structural_node_ids() {
        for extension in GRAPH_INTEGER_EXTENSIONS
            .iter()
            .chain(DIGRAPH_INTEGER_EXTENSIONS.iter())
        {
            assert!(extension.validate_error_contract().is_ok());
        }
        assert_eq!(GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert!(GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES[2]
            .inputs
            .iter()
            .all(|input| input.availability == BuiltinIntegerInputAvailability::Documented));
    }
}
