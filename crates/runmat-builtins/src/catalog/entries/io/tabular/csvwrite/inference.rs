use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let bytes = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![bytes]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !matches!(request.arguments.len(), 2 | 4) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVWRITE-ARITY",
            "csvwrite requires filename and data, with an optional row-col pair",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !scalar_text(value))
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVWRITE-PATH",
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
            "RM-CATALOG-CSVWRITE-PATH",
            "filename must not be empty",
            0,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(definitely_invalid_data)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CSVWRITE-DATA",
            "data must be a numeric or logical 2-D matrix",
            1,
        ));
    }
    for index in 2..request.arguments.len().min(4) {
        if definitely_invalid_offset(&request.arguments[index])
            || request
                .literals
                .literal_args
                .get(index)
                .is_some_and(invalid_literal_offset)
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-CSVWRITE-OFFSET",
                "row and col must be nonnegative integral scalar values",
                index,
            ));
        }
    }
    result
}

fn scalar_text(value: &ValueFact) -> bool {
    super::super::super::repl_fs::text_input::scalar(value)
}

fn definitely_invalid_data(value: &ValueFact) -> bool {
    if value
        .shape
        .known_dims()
        .is_some_and(|dims| dims.iter().skip(2).any(|dim| dim.is_some_and(|n| n != 1)))
    {
        return true;
    }
    !matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    )
}

fn definitely_invalid_offset(value: &ValueFact) -> bool {
    if value.shape.is_definitely_nonscalar() {
        return true;
    }
    match value.kind {
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical | ValueKindFact::Unknown => false,
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
