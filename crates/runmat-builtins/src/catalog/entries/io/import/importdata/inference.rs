use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut payload = ValueFact::unknown(DynamicReason::ExternalData);
    payload.residency = ResidencyFact::Host;
    let delimiter = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let header_lines = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![payload, delimiter, header_lines]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !(1..=3).contains(&request.arguments.len()) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-IMPORTDATA-ARITY",
            "importdata requires a filename and at most two controls",
            0,
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        if definitely_invalid_text(&request.arguments[index]) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-IMPORTDATA-TEXT",
                "filename and delimiter must be scalar text",
                index,
            ));
        }
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|text| text.trim().is_empty())
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-IMPORTDATA-EMPTY",
            "filename must not be empty",
            0,
        ));
    }
    if let Some(header) = request.arguments.get(2) {
        if definitely_invalid_header(header) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-IMPORTDATA-HEADER",
                "headerlinesIn must be a nonnegative real numeric scalar",
                2,
            ));
        }
    }
    if let Some(LiteralValue::Number(value)) = request.literals.literal_args.get(2) {
        if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-IMPORTDATA-HEADER",
                "headerlinesIn must be a nonnegative integer",
                2,
            ));
        }
    }
    result
}

fn definitely_invalid_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims.iter().skip(2).any(|dim| dim.is_some_and(|n| n != 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn definitely_invalid_header(value: &ValueFact) -> bool {
    if value.shape.is_definitely_nonscalar() {
        return true;
    }
    !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}
