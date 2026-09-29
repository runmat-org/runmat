//! Shared graph identity facts and error vocabulary.
//!
//! Each public graph operation owns its descriptor and documentation. This
//! module only contains the common graph object test and output families.

use crate::{
    catalog::inference::{argument_error, finish_fixed, finish_fixed_outputs},
    BuiltinCatalogEntry, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

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

#[derive(Clone, Copy)]
pub enum GraphCallKind {
    Adjacency,
    Bfsearch,
    Conncomp,
    Degree,
    Dfsearch,
    Digraph,
    Distances,
    Findedge,
    Graph,
    Indegree,
    Neighbors,
    Numedges,
    Numnodes,
    Outdegree,
    Predecessors,
    Shortestpath,
    Successors,
    Toposort,
    Treelayout,
}

pub fn infer_graph_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: GraphCallKind,
) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    let (minimum, maximum) = match kind {
        GraphCallKind::Graph | GraphCallKind::Digraph => (0, 4),
        GraphCallKind::Distances => (1, 3),
        GraphCallKind::Findedge | GraphCallKind::Shortestpath => (3, 3),
        GraphCallKind::Bfsearch
        | GraphCallKind::Dfsearch
        | GraphCallKind::Neighbors
        | GraphCallKind::Predecessors
        | GraphCallKind::Successors => (2, 2),
        _ => (1, 1),
    };
    if args.len() < minimum || args.len() > maximum {
        diagnostics.push(argument_error(
            "RM-CATALOG-GRAPH-ARITY",
            "invalid number of graph arguments",
            args.len().saturating_sub(1),
        ));
    }

    if matches!(kind, GraphCallKind::Graph | GraphCallKind::Digraph) {
        if args.len() == 1 && !possibly_square_adjacency(&args[0]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GRAPH-ADJACENCY",
                "one-argument graph construction requires a square adjacency array",
                0,
            ));
        }
        if args.len() >= 2 {
            let adjacency_with_labels = proven_square_adjacency(&args[0])
                && matches!(
                    args[1].kind,
                    ValueKindFact::Character | ValueKindFact::String
                );
            if adjacency_with_labels && args.len() > 2 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GRAPH-ARITY",
                    "adjacency construction accepts at most one option or name list",
                    2,
                ));
            }
            if !adjacency_with_labels {
                for (index, value) in args.iter().take(2).enumerate() {
                    if !possibly_node_list(value) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-GRAPH-NODES",
                            "edge endpoints must be numeric node vectors or name lists",
                            index,
                        ));
                    }
                }
                if args.len() == 4 && !possibly_numeric_vector(&args[2]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-GRAPH-WEIGHTS",
                        "the third argument must contain edge weights when names are fourth",
                        2,
                    ));
                }
            }
        }
        let class = match kind {
            GraphCallKind::Digraph => DIGRAPH_CLASS,
            _ => GRAPH_CLASS,
        };
        let mut output = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(class.owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        );
        output.residency = ResidencyFact::Host;
        return finish_fixed(entry, request, output, diagnostics);
    }

    if !matches!(kind, GraphCallKind::Treelayout) {
        if let Some(graph) = args.first() {
            if !possibly_graph_object(graph, matches!(kind, GraphCallKind::Toposort)) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GRAPH-OBJECT",
                    "G must be a scalar graph or digraph object",
                    0,
                ));
            }
        }
        let selectors = match kind {
            GraphCallKind::Findedge | GraphCallKind::Shortestpath => 2,
            GraphCallKind::Bfsearch
            | GraphCallKind::Dfsearch
            | GraphCallKind::Neighbors
            | GraphCallKind::Predecessors
            | GraphCallKind::Successors => 1,
            _ => 0,
        };
        for (offset, value) in args.iter().skip(1).take(selectors).enumerate() {
            if !possibly_node_scalar(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GRAPH-NODE",
                    "node selector must designate one numeric id or name",
                    offset + 1,
                ));
            }
        }
        if matches!(kind, GraphCallKind::Distances) {
            for (offset, value) in args.iter().skip(1).take(2).enumerate() {
                if !possibly_node_list(value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-GRAPH-NODES",
                        "distance selectors must be numeric or named node lists",
                        offset + 1,
                    ));
                }
            }
        }
    } else if let Some(parents) = args.first() {
        if !possibly_numeric_vector(parents) {
            diagnostics.push(argument_error(
                "RM-CATALOG-TREELAYOUT-PARENT",
                "parent must be a numeric or logical vector",
                0,
            ));
        }
    }

    let double = |shape| {
        let storage = if shape == ShapeFact::Scalar {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        };
        let mut output = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            storage,
        );
        output.residency = ResidencyFact::Host;
        output
    };
    let node_list = || {
        let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
        output.shape = ShapeFact::from(vec![None, Some(1)]);
        output.residency = ResidencyFact::Host;
        output
    };
    match kind {
        GraphCallKind::Numnodes | GraphCallKind::Numedges | GraphCallKind::Findedge => {
            finish_fixed(entry, request, double(ShapeFact::Scalar), diagnostics)
        }
        GraphCallKind::Shortestpath => finish_fixed_outputs(
            entry,
            request,
            vec![node_list(), double(ShapeFact::Scalar)],
            diagnostics,
        ),
        GraphCallKind::Treelayout => finish_fixed_outputs(
            entry,
            request,
            vec![
                double(ShapeFact::from(vec![Some(1), None])),
                double(ShapeFact::from(vec![Some(1), None])),
            ],
            diagnostics,
        ),
        GraphCallKind::Bfsearch
        | GraphCallKind::Dfsearch
        | GraphCallKind::Neighbors
        | GraphCallKind::Predecessors
        | GraphCallKind::Successors
        | GraphCallKind::Toposort => finish_fixed(entry, request, node_list(), diagnostics),
        GraphCallKind::Conncomp => finish_fixed(
            entry,
            request,
            double(ShapeFact::from(vec![Some(1), None])),
            diagnostics,
        ),
        GraphCallKind::Degree | GraphCallKind::Indegree | GraphCallKind::Outdegree => finish_fixed(
            entry,
            request,
            double(ShapeFact::from(vec![None, Some(1)])),
            diagnostics,
        ),
        GraphCallKind::Adjacency | GraphCallKind::Distances => finish_fixed(
            entry,
            request,
            double(ShapeFact::from(vec![None, None])),
            diagnostics,
        ),
        GraphCallKind::Graph | GraphCallKind::Digraph => unreachable!(),
    }
}

fn possibly_node_scalar(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Character => value.shape.known_dims().is_none_or(|dims| {
            dims.first()
                .is_none_or(|rows| rows.is_none_or(|rows| rows == 1))
        }),
        ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::String
        | ValueKindFact::Cell(_) => value.shape.element_count().is_none_or(|count| count == 1),
        _ => false,
    }
}

fn possibly_node_list(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Character => true,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => possibly_vector_shape(&value.shape),
        ValueKindFact::String | ValueKindFact::Cell(_) => possibly_vector_shape(&value.shape),
        _ => false,
    }
}

fn possibly_numeric_vector(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical
    ) && possibly_vector_shape(&value.shape)
}

fn possibly_vector_shape(shape: &ShapeFact) -> bool {
    let Some(dims) = shape.known_dims() else {
        return true;
    };
    dims.len() <= 2
        && (dims.len() < 2
            || dims[0].is_some_and(|rows| rows <= 1)
            || dims[1].is_some_and(|cols| cols <= 1)
            || dims[0].is_none()
            || dims[1].is_none())
}

fn possibly_graph_object(value: &ValueFact, directed_only: bool) -> bool {
    if value.shape.element_count().is_some_and(|count| count != 1)
        || matches!(value.residency, ResidencyFact::Remote { .. })
    {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => {
            let graph = GRAPH_CLASS.owned();
            let digraph = DIGRAPH_CLASS.owned();
            match object.runtime_class.as_ref() {
                Some(class) => class == &digraph || (!directed_only && class == &graph),
                None => object.class.is_none(),
            }
        }
        _ => false,
    }
}

fn possibly_square_adjacency(value: &ValueFact) -> bool {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) {
        return false;
    }
    match value.shape.known_dims() {
        Some(dims) if dims.len() == 2 => match (dims[0], dims[1]) {
            (Some(rows), Some(cols)) => rows == cols,
            _ => true,
        },
        Some(_) => false,
        None => true,
    }
}

fn proven_square_adjacency(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Numeric(_))
        && value.storage == StorageFact::Dense
        && value.shape.known_dims().is_some_and(|dims| {
            dims.len() == 2
                && matches!((dims[0], dims[1]), (Some(rows), Some(cols)) if rows == cols)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinIntegerInputAvailability;
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount, ValueKindFact};

    fn infer(
        entry: &BuiltinCatalogEntry,
        arguments: Vec<ValueFact>,
        outputs: usize,
    ) -> CallInference {
        crate::infer_catalog_call(
            entry,
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
            },
        )
    }

    fn graph_object(class: StaticClassIdentity) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(class.owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        )
    }

    #[test]
    fn construction_proves_graph_class_and_queries_keep_host_shapes() {
        let graph = infer(&super::super::GRAPH_CATALOG_ENTRY, vec![], 1);
        assert!(graph.diagnostics.is_empty(), "{:?}", graph.diagnostics);
        let ValueKindFact::Object(ref object) = graph.outputs[0].kind else {
            panic!("expected graph object");
        };
        assert_eq!(
            object.runtime_class.as_ref(),
            Some(&StaticClassIdentity::new("graph").owned())
        );
        let count = infer(
            &super::super::super::numnodes::NUMNODES_CATALOG_ENTRY,
            vec![graph_object(StaticClassIdentity::new("graph"))],
            1,
        );
        assert!(count.diagnostics.is_empty(), "{:?}", count.diagnostics);
        assert_eq!(count.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(count.outputs[0].residency, ResidencyFact::Host);
    }

    #[test]
    fn known_other_class_and_undirected_toposort_are_diagnosed() {
        let wrong = graph_object(StaticClassIdentity::new("tf"));
        let result = infer(
            &super::super::super::degree::DEGREE_CATALOG_ENTRY,
            vec![wrong],
            1,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-OBJECT"));
        let result = infer(
            &super::super::super::toposort::TOPOSORT_CATALOG_ENTRY,
            vec![graph_object(StaticClassIdentity::new("graph"))],
            1,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-OBJECT"));
    }

    #[test]
    fn two_result_forms_admit_exact_two_and_reject_three() {
        let graph = graph_object(StaticClassIdentity::new("digraph"));
        let endpoint = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let args = vec![graph, endpoint.clone(), endpoint];
        let two = infer(
            &super::super::super::shortestpath::SHORTESTPATH_CATALOG_ENTRY,
            args.clone(),
            2,
        );
        assert!(two.diagnostics.is_empty(), "{:?}", two.diagnostics);
        assert_eq!(two.outputs.len(), 2);
        let three = infer(
            &super::super::super::shortestpath::SHORTESTPATH_CATALOG_ENTRY,
            args,
            3,
        );
        assert!(!three.diagnostics.is_empty());
    }

    #[test]
    fn unresolved_constructor_role_does_not_become_a_false_adjacency_error() {
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::Scalar,
            StorageFact::Scalar,
        );
        let result = infer(
            &super::super::GRAPH_CATALOG_ENTRY,
            vec![unknown.clone(), text.clone(), unknown],
            1,
        );
        assert!(!result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-ARITY"));
        let wrong = infer(
            &super::super::GRAPH_CATALOG_ENTRY,
            vec![text.clone(), text],
            1,
        );
        assert!(wrong.diagnostics.is_empty(), "{:?}", wrong.diagnostics);
    }

    #[test]
    fn integer_graph_forms_separate_data_extensions_from_structural_node_ids() {
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
