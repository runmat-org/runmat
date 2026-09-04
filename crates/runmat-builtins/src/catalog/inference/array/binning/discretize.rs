use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, literal_text, support};

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
    output.residency = ResidencyFact::Host;

    let mut edges = double_fact(ShapeFact::from(vec![Some(1), None]));
    support::facts::materialize(&mut edges);
    edges.residency = ResidencyFact::Host;

    if request.outputs.requested.known_count() == Some(2) && !may_compute_edges(request) {
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
    if request.arguments.len() > 5 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DISCRETIZE-ARITY",
            "discretize accepts at most five inputs",
            5,
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

fn may_compute_edges(request: &CallRequest) -> bool {
    request
        .literals
        .numeric_at(1)
        .is_some_and(|value| value.is_finite() && value > 0.0 && value.fract() == 0.0)
        || request
            .arguments
            .get(1)
            .is_some_and(|value| value.is_scalar() && value.numeric().is_some())
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
