use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-COMPOSE-ARITY",
            "compose requires formatSpec",
            0,
        ));
    }
    let output = match request.arguments.first() {
        Some(input) if matches!(input.kind, ValueKindFact::String) => {
            let shape = if request.arguments.len() == 1 {
                input.shape.clone()
            } else {
                ShapeFact::Unknown
            };
            ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense)
        }
        Some(input)
            if matches!(
                input.kind,
                ValueKindFact::Character | ValueKindFact::Cell(_)
            ) =>
        {
            let shape =
                if request.arguments.len() == 1 && matches!(input.kind, ValueKindFact::Cell(_)) {
                    input.shape.clone()
                } else {
                    ShapeFact::Unknown
                };
            ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(ValueFact::proven(
                        ValueKindFact::Character,
                        ShapeFact::Unknown,
                        StorageFact::Dense,
                    )),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                shape,
                StorageFact::Dense,
            )
        }
        Some(input) if matches!(input.kind, ValueKindFact::Unknown) => {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COMPOSE-FORMAT",
                "compose requires text formatSpec",
                0,
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
