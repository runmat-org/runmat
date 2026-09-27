use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::Struct(_)
        ) || input.shape.element_count().is_some_and(|n| n != 1)
        {
            reject(&mut diagnostics, 0, "expected a scalar structure");
        }
    }

    finish(
        request,
        entry,
        vec![ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(vec![
                Some(1),
                request.arguments.first().and_then(|input| {
                    if let ValueKindFact::Struct(fields) = &input.kind {
                        fields
                            .fields_complete
                            .then(|| fields.fields.len().checked_mul(2))
                            .flatten()
                    } else {
                        None
                    }
                }),
            ]),
            StorageFact::Dense,
        )],
        diagnostics,
    )
}
