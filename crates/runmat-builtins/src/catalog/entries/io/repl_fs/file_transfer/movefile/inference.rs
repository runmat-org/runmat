use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs::file_transfer) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MOVEFILE-ARITY",
            "movefile expects source, destination, and an optional force flag",
            request.arguments.len().min(3),
        ));
    }
    for (index, label) in [(0, "source"), (1, "destination")] {
        if request
            .arguments
            .get(index)
            .is_some_and(|value| !super::super::validation::is_text_scalar(value))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MOVEFILE-TEXT",
                format!("movefile {label} must be a character row or string scalar"),
                index,
            ));
        }
    }
    if let Some(flag) = request.arguments.get(2) {
        let valid = request
            .literals
            .literal_string_at(2)
            .is_none_or(|text| text.eq_ignore_ascii_case("f"));
        if !super::super::validation::is_text_scalar(flag) || !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-MOVEFILE-FLAG",
                "movefile accepts only the force flag 'f'",
                2,
            ));
        }
    }
    finish_fixed_outputs(entry, request, super::super::facts::outputs(), diagnostics)
}
