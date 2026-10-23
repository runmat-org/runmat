use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-WINQUERYREG-ARITY",
            "winqueryreg requires two or three controls",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !super::super::text_input::scalar(argument)
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-WINQUERYREG-TEXT",
                "registry selectors must be host scalar text",
                index,
            ));
        }
    }
    let names = matches!(request.literals.literal_args.first(), Some(LiteralValue::String(name) | LiteralValue::Character(name)) if name == "name")
        && request.arguments.len() == 3;
    let result = if names {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                )),
                elements: vec![],
                elements_complete: false,
            }),
            ShapeFact::from(vec![None, Some(1)]),
            StorageFact::Dense,
        )
    } else {
        // Registry payload type is external state. Do not invent text or double
        // facts: REG_DWORD returns an exact signed int32 bit interpretation.
        let mut result = ValueFact::unknown(DynamicReason::ExternalData);
        result.residency = ResidencyFact::Host;
        result
    };
    finish_fixed_outputs(entry, request, vec![result], diagnostics)
}
