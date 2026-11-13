use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

use super::precision::{parse_fread_precision, FreadOutputKind};

#[derive(Clone, Copy)]
enum ProvenOutput {
    Numeric(NumericClass),
    Character,
    Dynamic,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let data = match proven_output(request) {
        ProvenOutput::Numeric(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Ranked { rank: 2 },
            StorageFact::Dense,
        ),
        ProvenOutput::Character => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::Ranked { rank: 2 },
            StorageFact::Dense,
        ),
        ProvenOutput::Dynamic => ValueFact::unknown(DynamicReason::DynamicDispatch),
    };
    let count = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![data, count.clone()]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(count));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.is_empty() || request.arguments.len() > 7 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FREAD-ARITY",
            "fread requires a file identifier and admits at most four controls plus a Like pair",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(definitely_invalid_fid)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FREAD-FID",
            "file identifier must be a real numeric or logical scalar",
            0,
        ));
    }
    result
}

fn proven_output(request: &CallRequest) -> ProvenOutput {
    if request.arguments.len() == 1 {
        return ProvenOutput::Numeric(NumericClass::Double);
    }
    // An unresolved control or Like prototype may change the output class.
    if request
        .arguments
        .iter()
        .skip(1)
        .any(|arg| matches!(&arg.kind, ValueKindFact::Unknown))
        || (1..request.arguments.len()).any(|index| {
            request
                .literals
                .literal_string_at(index)
                .is_some_and(|text| text.eq_ignore_ascii_case("like"))
        })
    {
        return ProvenOutput::Dynamic;
    }
    let precision_index = if matches!(
        request.arguments.get(1).map(|a| &a.kind),
        Some(ValueKindFact::String | ValueKindFact::Character)
    ) {
        Some(1)
    } else if matches!(
        request.arguments.get(2).map(|a| &a.kind),
        Some(ValueKindFact::String | ValueKindFact::Character)
    ) {
        Some(2)
    } else {
        None
    };
    let Some(index) = precision_index else {
        return ProvenOutput::Numeric(NumericClass::Double);
    };
    if (1..request.arguments.len()).any(|other| {
        other != index
            && matches!(
                &request.arguments[other].kind,
                ValueKindFact::String | ValueKindFact::Character
            )
            && request.literals.literal_string_at(other).is_none()
    }) {
        return ProvenOutput::Dynamic;
    }
    let Some(label) = request.literals.literal_string_at(index) else {
        return ProvenOutput::Dynamic;
    };
    match parse_fread_precision(&label) {
        Ok(spec) => match spec.output {
            FreadOutputKind::Numeric(class) => ProvenOutput::Numeric(class),
            FreadOutputKind::Char => ProvenOutput::Character,
        },
        Err(_) => ProvenOutput::Dynamic,
    }
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
