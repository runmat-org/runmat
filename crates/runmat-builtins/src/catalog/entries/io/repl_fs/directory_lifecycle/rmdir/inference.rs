use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs::directory_lifecycle) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    validate_arity(request, &mut diagnostics);
    if let Some(folder) = request.arguments.first() {
        if !super::super::validation::is_text_scalar(folder) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMDIR-TEXT",
                "rmdir expects a character-row or string-scalar folder name",
                0,
            ));
        }
    }
    validate_options(request, &mut diagnostics);
    finish_fixed_outputs(entry, request, super::super::facts::outputs(), diagnostics)
}

fn validate_arity(request: &CallRequest, diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>) {
    if !(1..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RMDIR-ARITY",
            "rmdir expects a folder, optional 's', and optional ResolveSymbolicLinks value",
            request.arguments.len().min(4),
        ));
    }
}

fn validate_options(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(second) = request.arguments.get(1) else {
        return;
    };
    let second_literal = request.literals.literal_string_at(1);
    let option_index = match second_literal.as_deref() {
        Some(text) if text.eq_ignore_ascii_case("s") => 2,
        Some(text) if text.eq_ignore_ascii_case("ResolveSymbolicLinks") => 1,
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMDIR-OPTION",
                "rmdir accepts only 's' or ResolveSymbolicLinks after the folder",
                1,
            ));
            return;
        }
        None if !super::super::validation::is_text_scalar(second) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMDIR-OPTION",
                "rmdir options must be character rows or string scalars",
                1,
            ));
            return;
        }
        None => return,
    };
    validate_symbolic_link_option(request, option_index, diagnostics);
}

fn validate_symbolic_link_option(
    request: &CallRequest,
    option_index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(_) = request.arguments.get(option_index) else {
        return;
    };
    let is_symbolic_link_option = request
        .literals
        .literal_string_at(option_index)
        .is_some_and(|name| name.eq_ignore_ascii_case("ResolveSymbolicLinks"));
    if !is_symbolic_link_option {
        if option_index > 1 || request.arguments.len() > option_index + 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMDIR-OPTION-NAME",
                "rmdir expects the ResolveSymbolicLinks option name",
                option_index,
            ));
        }
        return;
    }
    let value_index = option_index + 1;
    match request.arguments.get(value_index) {
        Some(value) if super::super::validation::is_logical_control(value) => {}
        Some(_) => diagnostics.push(argument_error(
            "RM-CATALOG-RMDIR-OPTION-VALUE",
            "ResolveSymbolicLinks must be a logical or exact numeric scalar zero or one",
            value_index,
        )),
        None => diagnostics.push(argument_error(
            "RM-CATALOG-RMDIR-OPTION-VALUE",
            "ResolveSymbolicLinks requires a value",
            value_index,
        )),
    }
}
