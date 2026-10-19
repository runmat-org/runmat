use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use super::super::super::{argument_error, support};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = validate(request);
    let mut output = output_fact(request);
    output.shape = output_shape(request);
    support::facts::materialize_preserving_sparse_storage(&mut output);
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn validate(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if !(2..=6).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ACCUMARRAY-ARITY",
            "accumarray requires two through six inputs",
            request.arguments.len().min(6),
        ));
    }
    if let Some(function) = request.arguments.get(3) {
        let omitted = matches!(
            request.literals.literal_args.get(3),
            Some(LiteralValue::Empty)
        );
        if !omitted
            && !matches!(
                function.kind,
                ValueKindFact::Callable(_) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ACCUMARRAY-FUNCTION",
                "accumarray requires a callable group function or []",
                3,
            ));
        }
    }
    diagnostics
}

fn output_fact(request: &CallRequest) -> ValueFact {
    let mut output = callback_output(request).unwrap_or_else(|| double_output(ShapeFact::Unknown));
    output.storage = match sparse_control(request) {
        Some(true) => StorageFact::Sparse,
        Some(false) => StorageFact::Dense,
        None if request.arguments.len() >= 6 => StorageFact::Unknown,
        None => StorageFact::Dense,
    };
    output
}

fn callback_output(request: &CallRequest) -> Option<ValueFact> {
    if request.arguments.len() < 4
        || matches!(
            request.literals.literal_args.get(3),
            Some(LiteralValue::Empty)
        )
    {
        return None;
    }
    match &request.arguments.get(3)?.kind {
        ValueKindFact::Callable(callable) => callable.outputs.first().cloned(),
        _ => Some(ValueFact::unknown(DynamicReason::UnresolvedCallable)),
    }
}

fn output_shape(request: &CallRequest) -> ShapeFact {
    if let Some(mut dimensions) = request.literals.numeric_vector_at(2) {
        if dimensions.len() == 1 {
            dimensions.push(Some(1));
        }
        return ShapeFact::from(dimensions);
    }
    let Some(indices) = request.arguments.first() else {
        return ShapeFact::Unknown;
    };
    let rank = match indices.shape.known_dims() {
        Some(dimensions) if dimensions.get(1).copied().flatten().unwrap_or(1) > 1 => dimensions[1],
        Some(_) => Some(1),
        None => None,
    };
    rank.map_or(ShapeFact::Unknown, |rank| {
        ShapeFact::from(vec![None; rank.max(2)])
    })
}

fn sparse_control(request: &CallRequest) -> Option<bool> {
    request.literals.literal_bool_at(5).or_else(|| {
        request
            .literals
            .numeric_at(5)
            .and_then(|value| match value {
                0.0 => Some(false),
                1.0 => Some(true),
                _ => None,
            })
    })
}

fn double_output(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
