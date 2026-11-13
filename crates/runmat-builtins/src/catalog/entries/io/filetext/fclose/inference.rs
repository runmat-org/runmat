use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(vec![double(), character_row()]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() > 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FCLOSE-ARITY",
            "fclose accepts at most one argument",
            1,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(definitely_invalid_input)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FCLOSE-FID",
            "file identifiers must be numeric, logical, a nested collection, or the all keyword",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|text| !text.eq_ignore_ascii_case("all"))
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FCLOSE-ALL",
            "the only text selector is all",
            0,
        ));
    }
    result
}

fn definitely_invalid_input(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical | ValueKindFact::Unknown => false,
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|dim| dim != 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Cell(_) => false,
        _ => true,
    }
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}
