use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DimensionFact,
    DynamicReason, LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Interp1Form {
    Implicit,
    Explicit,
    Dynamic,
}

fn form(request: &CallRequest) -> Interp1Form {
    if request.arguments.len() == 2 {
        return Interp1Form::Implicit;
    }
    if matches!(
        request.literals.literal_args.get(2),
        Some(LiteralValue::Character(_) | LiteralValue::String(_) | LiteralValue::Keyword(_))
    ) {
        return Interp1Form::Implicit;
    }
    let Some(third) = request.arguments.get(2) else {
        return Interp1Form::Dynamic;
    };
    match third.kind {
        ValueKindFact::Character | ValueKindFact::String => Interp1Form::Implicit,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => Interp1Form::Explicit,
        _ => Interp1Form::Dynamic,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=5).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTERP1-ARITY",
            "interp1 expects sample values and query points, with optional locations and options",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let (value_index, query_index) = match form(request) {
        Interp1Form::Implicit => (Some(0), Some(1)),
        Interp1Form::Explicit => (Some(1), Some(2)),
        Interp1Form::Dynamic => (None, None),
    };
    for index in [value_index, query_index].into_iter().flatten() {
        if let Some(value) = request.arguments.get(index) {
            if !matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INTERP1-NUMERIC",
                    "interp1 samples and query points must be real numeric or logical",
                    index,
                ));
            }
        }
    }
    let output = match form(request) {
        Interp1Form::Implicit => result_fact(request, 0, 1),
        Interp1Form::Explicit => result_fact(request, 1, 2),
        Interp1Form::Dynamic => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn result_fact(request: &CallRequest, values_index: usize, query_index: usize) -> ValueFact {
    let (Some(values), Some(query)) = (
        request.arguments.get(values_index),
        request.arguments.get(query_index),
    ) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = output_shape(&values.shape, &query.shape);
    if !matches!(
        values.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical
    ) {
        let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
        output.shape = shape;
        return output;
    }
    // Host evaluation uses binary64; the resident linear/nearest path can
    // retain F32. Method and provider availability are runtime decisions, so
    // do not promise a host class or residency for resident sample values.
    if values.residency != ResidencyFact::Host {
        let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
        output.shape = shape;
        return output;
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Unknown,
    );
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    // Query coordinates may be resident even when samples are host values;
    // that path gathers the query and returns host binary64.
    output
}

fn output_shape(values: &ShapeFact, query: &ShapeFact) -> ShapeFact {
    if super::super::piecewise_inference::known_vector(values) {
        return query.clone();
    }
    let ShapeFact::Shaped { dims } = values else {
        return ShapeFact::Unknown;
    };
    if dims.len() < 2 {
        return ShapeFact::Unknown;
    }
    let series_count = dims[1..].iter().try_fold(1usize, |count, dim| {
        let DimensionFact::Known(extent) = dim else {
            return None;
        };
        count.checked_mul(*extent)
    });
    if series_count == Some(1) {
        return query.clone();
    }
    if series_count.is_none() {
        return ShapeFact::Unknown;
    }
    let mut output = match query {
        ShapeFact::Shaped { dims } => dims.clone(),
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        _ => return ShapeFact::Unknown,
    };
    output.extend_from_slice(&dims[1..]);
    ShapeFact::Shaped { dims: output }
}
