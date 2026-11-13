use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    for (index, argument) in request.arguments.iter().enumerate() {
        // Only the leading argument can be a numeric axes identity. Later
        // values may be labels or style property values, whose exact roles
        // require the runtime parser and live graphics state.
        let invalid = if index == 0 && matches!(argument.kind, ValueKindFact::Numeric(_)) {
            argument
                .numeric()
                .is_some_and(|number| number.domain == NumericDomain::Complex)
                || argument
                    .shape
                    .element_count()
                    .is_some_and(|count| count != 1)
                || argument.storage == StorageFact::Sparse
        } else if index == 0 {
            !matches!(
                argument.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Character
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
            )
        } else {
            matches!(argument.kind, ValueKindFact::Void | ValueKindFact::Never)
        };
        if invalid
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LEGEND-ARGUMENT",
                "legend requires host axes, labels, commands, and style values",
                index,
            ));
        }
    }
    // A handle is a live graphics identity, not a statically known number.
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}
