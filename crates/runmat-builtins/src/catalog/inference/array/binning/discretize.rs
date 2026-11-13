use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, literal_text, support};

mod validation;

use validation::validate_tail;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = validate_request(request);
    let output_shape = request
        .arguments
        .first()
        .map(|value| value.shape.clone())
        .unwrap_or(ShapeFact::Unknown);
    let mut output = replacement_fact(request).unwrap_or_else(|| double_fact(output_shape.clone()));
    output.shape = output_shape;
    support::facts::materialize(&mut output);
    // This builtin always constructs a Tensor/StringArray, even for one element.
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;

    let edge_count = request
        .literals
        .numeric_at(1)
        .filter(|count| {
            count.is_finite() && *count > 0.0 && count.fract() == 0.0 && *count < usize::MAX as f64
        })
        .and_then(|count| (count as usize).checked_add(1));
    let mut edges = double_fact(ShapeFact::from(vec![Some(1), edge_count]));
    support::facts::materialize(&mut edges);
    edges.residency = ResidencyFact::Host;

    if request.outputs.requested.known_count() == Some(2)
        && request
            .arguments
            .get(1)
            .and_then(|value| value.shape.element_count())
            .is_some_and(|count| count != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DISCRETIZE-OUTPUTS",
            "discretize returns a second edge output only for a scalar bin count",
            1,
        ));
    }

    let mut contract = CallContract::fixed(vec![output, edges]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn validate_request(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DISCRETIZE-ARITY",
            "discretize requires X and numeric edges or a scalar bin count",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        let supported = matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
        );
        if !supported && !matches!(input.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISCRETIZE-X",
                "discretize currently requires real numeric or logical X",
                0,
            ));
        }
    }
    if let Some(edge) = request.arguments.get(1) {
        if !matches!(
            edge.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISCRETIZE-EDGES",
                "discretize requires real numeric edges or a bin count",
                1,
            ));
        }
        if known_nonvector(&edge.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISCRETIZE-EDGES",
                "discretize edges must be a vector",
                1,
            ));
        }
    }
    if let Some(count) = request.literals.numeric_at(1) {
        if !count.is_finite() || count <= 0.0 || count.fract() != 0.0 {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISCRETIZE-COUNT",
                "discretize bin count is outside the supported positive integer range",
                1,
            ));
        }
    }
    validate_tail(request, &mut diagnostics);
    diagnostics
}

fn replacement_fact(request: &CallRequest) -> Option<ValueFact> {
    let replacement = request.arguments.get(2)?;
    if request
        .literals
        .literal_args
        .get(2)
        .and_then(literal_text)
        .is_some_and(|name| name.eq_ignore_ascii_case("IncludedEdge"))
    {
        return None;
    }
    if request.arguments.len() >= 4
        && request
            .literals
            .literal_args
            .get(2)
            .and_then(literal_text)
            .is_none()
        && matches!(
            replacement.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        )
    {
        return Some(ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    let kind = match &replacement.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            ValueKindFact::Numeric(*numeric)
        }
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_) => {
            ValueKindFact::String
        }
        ValueKindFact::Unknown => return Some(ValueFact::unknown(DynamicReason::RuntimeValue)),
        _ => return Some(ValueFact::unknown(DynamicReason::UnsupportedRepresentation)),
    };
    Some(ValueFact::proven(
        kind,
        ShapeFact::Unknown,
        StorageFact::Dense,
    ))
}

pub(super) fn known_nonvector(shape: &ShapeFact) -> bool {
    let Some(dims) = shape.known_dims() else {
        return false;
    };
    dims.iter()
        .skip(2)
        .any(|extent| extent.is_some_and(|extent| extent != 1))
        || dims
            .iter()
            .take(2)
            .filter(|extent| extent.is_some_and(|extent| extent != 1))
            .count()
            > 1
}

fn double_fact(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
