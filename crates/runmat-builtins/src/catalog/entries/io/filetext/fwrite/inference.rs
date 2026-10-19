use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !(2..=5).contains(&request.arguments.len()) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FWRITE-ARITY",
            "fwrite requires a file identifier and data, with at most three controls",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(definitely_invalid_fid)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FWRITE-FID",
            "file identifier must be a real numeric or logical scalar",
            0,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(definitely_invalid_data)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FWRITE-DATA",
            "data must be real numeric, logical, or text",
            1,
        ));
    }
    if request
        .arguments
        .get(2)
        .is_some_and(definitely_invalid_text_control)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FWRITE-PRECISION",
            "precision must be a string scalar or character row",
            2,
        ));
    }
    result
}

fn definitely_invalid_fid(value: &ValueFact) -> bool {
    if value.shape.is_definitely_nonscalar() {
        return true;
    }
    match &value.kind {
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical | ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn definitely_invalid_data(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn definitely_invalid_text_control(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => match &value.shape {
            ShapeFact::Shaped { dims } => {
                matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1)
                    || dims
                        .iter()
                        .skip(2)
                        .any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1))
            }
            _ => false,
        },
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        _ => true,
    }
}
