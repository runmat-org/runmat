use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(progress) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-WAITBAR-ARITY",
            "waitbar requires progress",
            0,
        ));
        return finish_fixed(entry, request, handle_fact(), diagnostics);
    };
    let invalid_progress = match progress.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Complex,
        _ => true,
    } || progress
        .shape
        .element_count()
        .is_some_and(|count| count != 1)
        || progress.storage == StorageFact::Sparse
        || matches!(
            progress.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        )
        || request
            .literals
            .numeric_at(0)
            .is_some_and(|value| !value.is_finite());
    if invalid_progress {
        diagnostics.push(argument_error(
            "RM-CATALOG-WAITBAR-PROGRESS",
            "progress must be a finite host real numeric scalar",
            0,
        ));
    }
    // A second numeric argument may be an existing figure handle; a text or
    // cell value may be a message. Remaining property values are parsed by
    // the figure owner, so avoid guessing their roles from their classes.
    if let Some(second) = request.arguments.get(1) {
        let invalid = match second.kind {
            ValueKindFact::Unknown
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Cell(_) => false,
            ValueKindFact::Numeric(number) => {
                number.domain == NumericDomain::Complex
                    || second.shape.element_count().is_some_and(|count| count != 1)
                    || second.storage == StorageFact::Sparse
            }
            _ => true,
        } || matches!(
            second.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        );
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-WAITBAR-SECOND",
                "expected host message text or a scalar figure handle",
                1,
            ));
        }
    }
    finish_fixed(entry, request, handle_fact(), diagnostics)
}

fn handle_fact() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
