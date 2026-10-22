use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISHANDLE-ARITY",
            "ishandle expects exactly one candidate",
            0,
        ));
    }

    let output = match request.arguments.first() {
        Some(input) if matches!(input.residency, ResidencyFact::Device { .. }) => {
            if input.shape.element_count() == Some(1) {
                ValueFact::scalar(ValueKindFact::Logical)
            } else {
                ValueFact::proven(
                    ValueKindFact::Logical,
                    input.shape.clone(),
                    StorageFact::Dense,
                )
            }
        }
        Some(input) if input.residency == ResidencyFact::Unknown => ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::Unknown,
            StorageFact::Unknown,
        ),
        Some(input)
            if input
                .numeric()
                .is_some_and(|n| n.domain == NumericDomain::Real)
                && matches!(input.storage, StorageFact::Scalar | StorageFact::Dense) =>
        {
            ValueFact::proven(ValueKindFact::Logical, input.shape.clone(), input.storage)
        }
        Some(input)
            if input
                .numeric()
                .is_some_and(|n| n.domain == NumericDomain::Real)
                && input.storage == StorageFact::Unknown =>
        {
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::Unknown,
                StorageFact::Unknown,
            )
        }
        Some(input) if input.kind != ValueKindFact::Unknown => {
            ValueFact::scalar(ValueKindFact::Logical)
        }
        _ => ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::Unknown,
            StorageFact::Unknown,
        ),
    };
    finish_fixed(entry, request, output, diagnostics)
}
