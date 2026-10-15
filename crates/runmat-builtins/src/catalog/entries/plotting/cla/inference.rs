use crate::{
    catalog::inference::{
        argument_error, finish_fixed_outputs, known_invalid_scalar_object_handle,
    },
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    let count = request.arguments.len();
    if count > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CLA-ARITY",
            "cla accepts an optional axes handle followed by an optional mode",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let target_possible = index == 0;
        let numeric_target = target_possible && matches!(argument.kind, ValueKindFact::Numeric(_));
        let invalid = if target_possible && matches!(argument.kind, ValueKindFact::Object(_)) {
            known_invalid_scalar_object_handle(argument, &[standard::AXES])
        } else if numeric_target {
            argument
                .numeric()
                .is_some_and(|n| n.domain == NumericDomain::Complex)
                || argument.shape.element_count().is_some_and(|n| n != 1)
                || matches!(argument.storage, StorageFact::Sparse)
        } else {
            match argument.kind {
                ValueKindFact::Unknown => false,
                ValueKindFact::Character => {
                    !crate::catalog::inference::possible_character_row(&argument.shape)
                }
                ValueKindFact::String => {
                    !matches!(argument.storage, StorageFact::Unknown | StorageFact::Scalar)
                }
                _ => true,
            }
        };
        let invalid_text = request
            .literals
            .literal_string_at(index)
            .is_some_and(|text| {
                !matches!(text.trim().to_ascii_lowercase().as_str(), "reset")
                    || (count == 2 && index == 0)
            });
        if invalid
            || invalid_text
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-CLA-ARGUMENT",
                "cla requires a host scalar axes handle or supported text mode",
                index,
            ));
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Logical);
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
