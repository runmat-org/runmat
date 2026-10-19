use super::super::inference::{definitely_non_text, finish, has_stored_shape, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        let option = request.literals.literal_string_at(1);
        if request
            .arguments
            .get(1)
            .is_some_and(|value| definitely_non_text(value, false))
        {
            reject(&mut diagnostics, 1, "expected a scalar text option");
        }
        if option
            .as_ref()
            .is_some_and(|text| !text.eq_ignore_ascii_case("allow-all-empties"))
        {
            reject(&mut diagnostics, 1, "option must be allow-all-empties");
        }
        if let Some(dims) = input.shape.known_dims().filter(|_| has_stored_shape(input)) {
            let definitely_nd = dims.iter().skip(2).any(|d| d.is_some_and(|n| n != 1));
            let definitely_matrix = dims.first().is_some_and(|d| d.is_some_and(|n| n != 1))
                && dims.get(1).is_some_and(|d| d.is_some_and(|n| n != 1));
            let may_allow_empty = request.arguments.len() == 2
                && option
                    .as_ref()
                    .is_none_or(|s| s.eq_ignore_ascii_case("allow-all-empties"));
            let proven_nonempty = input.shape.element_count().is_some_and(|n| n > 0);
            if (definitely_nd || definitely_matrix) && (!may_allow_empty || proven_nonempty) {
                reject(
                    &mut diagnostics,
                    0,
                    "expected a vector or an admitted empty shape",
                );
            }
        }
    }

    finish(request, entry, vec![], diagnostics)
}
