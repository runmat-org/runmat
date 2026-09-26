use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    let count = request.arguments.len();
    if count != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHADING-ARITY",
            "shading accepts exactly one mode",
            1,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let invalid = match argument.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Character => {
                !crate::catalog::inference::possible_character_row(&argument.shape)
            }
            ValueKindFact::String => {
                !matches!(argument.storage, StorageFact::Unknown | StorageFact::Scalar)
            }
            _ => true,
        };
        let invalid_text = request
            .literals
            .literal_string_at(index)
            .is_some_and(|text| {
                !matches!(
                    text.trim().to_ascii_lowercase().as_str(),
                    "flat" | "interp" | "faceted"
                )
            });
        if invalid
            || invalid_text
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SHADING-ARGUMENT",
                "shading requires supported text mode",
                index,
            ));
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Logical);
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
