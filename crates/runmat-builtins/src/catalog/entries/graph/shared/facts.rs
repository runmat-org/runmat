//! Reusable facts for graph inference. Identity-specific arity and output rules
//! remain beside each public builtin.

use super::{DIGRAPH_CLASS, GRAPH_CLASS};
use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(crate) fn arity(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    minimum: usize,
    maximum: usize,
) {
    let count = request.arguments.len();
    if count < minimum || count > maximum {
        diagnostics.push(argument_error(
            "RM-CATALOG-GRAPH-ARITY",
            "invalid number of graph arguments",
            count.saturating_sub(1),
        ));
    }
}

pub(crate) fn adjacency_argument(value: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if !possibly_square_adjacency(value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GRAPH-ADJACENCY",
            "one-argument graph construction requires a square adjacency array",
            0,
        ));
    }
}

pub(crate) fn constructor_arguments(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let args = &request.arguments;
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

pub(crate) fn graph_operand(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    directed_only: bool,
) {
    if let Some(graph) = request.arguments.first() {
        if !possibly_graph_object(graph, directed_only) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GRAPH-OBJECT",
                "G must be a scalar graph or digraph object",
                0,
            ));
        }
    }
}

pub(crate) fn node_selector(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
) {
    if let Some(value) = request.arguments.get(index) {
        if !possibly_node_scalar(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GRAPH-NODE",
                "node selector must designate one numeric id or name",
                index,
            ));
        }
    }
}

pub(crate) fn node_list_selector(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
) {
    if let Some(value) = request.arguments.get(index) {
        if !possibly_node_list(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GRAPH-NODES",
                "distance selectors must be numeric or named node lists",
                index,
            ));
        }
    }
}

pub(crate) fn parent_vector(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(parents) = request.arguments.first() {
        if !possibly_numeric_vector(parents) {
            diagnostics.push(argument_error(
                "RM-CATALOG-TREELAYOUT-PARENT",
                "parent must be a numeric or logical vector",
                0,
            ));
        }
    }
}

pub(crate) fn graph_object_fact(class: StaticClassIdentity) -> ValueFact {
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
    output
}

pub(crate) fn double_fact(shape: ShapeFact) -> ValueFact {
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
}

pub(crate) fn node_list_fact() -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::from(vec![None, Some(1)]);
    output.residency = ResidencyFact::Host;
    output
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
    matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical
    ) && value.storage == StorageFact::Dense
        && value.shape.known_dims().is_some_and(|dims| {
            dims.len() == 2
                && matches!((dims[0], dims[1]), (Some(rows), Some(cols)) if rows == cols)
        })
}
