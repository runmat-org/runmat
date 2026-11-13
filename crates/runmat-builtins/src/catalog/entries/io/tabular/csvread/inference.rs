use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut matrix = ValueFact::unknown(DynamicReason::ExternalData);
    matrix.shape = ShapeFact::from(vec![None, None]);
    matrix.storage = StorageFact::Dense;
    matrix.layout = LayoutFact::ColumnMajor;
    matrix.contiguity = ContiguityFact::Contiguous;
    matrix.view = ViewFact::Materialized;
    matrix.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![matrix]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);

    if !matches!(request.arguments.len(), 1 | 3 | 4) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVREAD-ARITY",
            "csvread requires filename, optional row and col, and optional range",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !scalar_text(value))
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVREAD-PATH",
            "filename must be scalar text",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|text| text.trim().is_empty())
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVREAD-PATH",
            "filename must not be empty",
            0,
        ));
    }
    for index in 1..request.arguments.len().min(3) {
        if definitely_invalid_offset(&request.arguments[index])
            || request
                .literals
                .literal_args
                .get(index)
                .is_some_and(invalid_literal_offset)
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-CSVREAD-OFFSET",
                "row and col must be nonnegative integral scalar values",
                index,
            ));
        }
    }
    if request
        .arguments
        .get(3)
        .is_some_and(definitely_invalid_range)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVREAD-RANGE",
            "range must be scalar text or a numeric vector",
            3,
        ));
    }
    result
}

fn scalar_text(value: &ValueFact) -> bool {
    super::super::super::repl_fs::text_input::scalar(value)
}

fn definitely_invalid_offset(value: &ValueFact) -> bool {
    if value.shape.is_definitely_nonscalar() {
        return true;
    }
    match value.kind {
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn invalid_literal_offset(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Number(n) => !n.is_finite() || *n < 0.0 || n.fract() != 0.0,
        LiteralValue::Integer { .. } => value.integer_value_i128().is_some_and(|n| n < 0),
        _ => false,
    }
}

fn definitely_invalid_range(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Character | ValueKindFact::String => !scalar_text(value),
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        _ => true,
    }
}
