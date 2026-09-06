use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs::path_syntax) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULLFILE-ARITY",
            "fullfile expects at least one path component",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let numeric_extension = super::super::validation::is_numeric_character_row(argument);
        if !super::super::validation::is_text_container(argument) && !numeric_extension {
            diagnostics.push(argument_error(
                "RM-CATALOG-FULLFILE-TEXT",
                "fullfile expects character rows, string arrays, or cells of character rows",
                index,
            ));
        }
    }
    if super::super::validation::has_known_shape_mismatch(&request.arguments) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULLFILE-SHAPE",
            "fullfile nonscalar string and cell inputs must have the same shape",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        super::super::facts::join_result(&request.arguments),
        diagnostics,
    )
}
