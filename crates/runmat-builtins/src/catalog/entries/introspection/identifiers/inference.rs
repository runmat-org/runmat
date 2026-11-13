use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer_namelengthmax(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    let mut result = infer_call(&contract, request);
    if !request.arguments.is_empty() {
        result.diagnostics.push(argument_error(
            "RM.NAMELENGTHMAX.ARG_COUNT",
            "expected no arguments",
            0,
        ));
    }
    result
}

pub(super) fn infer_iskeyword(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = if request.arguments.is_empty() {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new({
                    let mut element = ValueFact::unknown(DynamicReason::RuntimeValue);
                    element.kind = ValueKindFact::Character;
                    element.residency = ResidencyFact::Host;
                    element
                }),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(vec![None, Some(1)]),
            StorageFact::Dense,
        )
    } else {
        ValueFact::scalar(ValueKindFact::Logical)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() > 1 {
        result.diagnostics.push(argument_error(
            "RM.ISKEYWORD.ARG_COUNT",
            "expected zero or one argument",
            0,
        ));
    }
    if let Some(arg) = request.arguments.first() {
        if !matches!(
            arg.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || match arg.kind {
            ValueKindFact::Character => {
                !crate::catalog::inference::possible_character_row(&arg.shape)
            }
            _ => crate::catalog::inference::known_invalid_text_scalar(arg),
        } {
            result.diagnostics.push(argument_error(
                "RM.ISKEYWORD.TEXT",
                "expected text scalar",
                0,
            ));
        }
    }
    result
}
