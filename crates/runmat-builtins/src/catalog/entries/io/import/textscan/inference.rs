use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![cell]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TEXTSCAN-ARITY",
            "textscan requires text or a file identifier and a format specification",
            0,
        ));
    }
    if let Some(value) = request.arguments.first() {
        if definitely_invalid_input(value) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-TEXTSCAN-INPUT",
                "textscan input must be scalar text or a real numeric file identifier",
                0,
            ));
        }
        if matches!(value.kind, ValueKindFact::Character | ValueKindFact::String) {
            result
                .capabilities
                .0
                .remove(&CapabilityRequirement::Filesystem);
            result.effects.0.remove(&EffectKind::FilesystemRead);
        }
    }
    if request
        .arguments
        .get(1)
        .is_some_and(definitely_invalid_text)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TEXTSCAN-FORMAT",
            "formatSpec must be scalar text",
            1,
        ));
    }
    if request.arguments.len() > 2 {
        let tail = request.arguments.len() - 2;
        let definitely_bad_parity = match request.arguments.get(2).map(|value| &value.kind) {
            Some(ValueKindFact::Numeric(_)) => tail.is_multiple_of(2),
            Some(ValueKindFact::Character | ValueKindFact::String) => !tail.is_multiple_of(2),
            _ => false,
        };
        if definitely_bad_parity {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-TEXTSCAN-OPTIONS",
                "textscan options require name-value pairs after an optional repeat count",
                2,
            ));
        }
    }
    result
}

fn definitely_invalid_input(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Character | ValueKindFact::String => definitely_invalid_text(value),
        ValueKindFact::Numeric(fact) => {
            fact.domain == NumericDomain::Complex || value.shape.is_definitely_nonscalar()
        }
        _ => true,
    }
}

fn definitely_invalid_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|extent| extent.is_some_and(|n| n != 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}
