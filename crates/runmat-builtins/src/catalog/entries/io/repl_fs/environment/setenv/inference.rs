use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use super::super::{facts, validation};

pub(in crate::catalog::entries::io::repl_fs::environment) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SETENV-ARITY",
            "setenv expects one or two inputs",
            request.arguments.len().min(2),
        ));
    }
    if let Some(name) = request.arguments.first() {
        let valid = if request.arguments.len() == 1 {
            validation::valid_name_or_dictionary(name)
        } else {
            validation::valid_name(name, validation::NamePolicy::Matlab)
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SETENV-NAME",
                "setenv expects an environment dictionary or supported text names",
                0,
            ));
        }
    }
    if let Some(value) = request.arguments.get(1) {
        if !validation::valid_value(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SETENV-VALUE",
                "setenv values must be text, scalar numeric values, or matching containers",
                1,
            ));
        } else if request
            .arguments
            .first()
            .is_some_and(|name| !validation::shapes_are_compatible(name, value))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SETENV-SHAPE",
                "setenv name and value containers must have matching shapes",
                1,
            ));
        }
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![facts::double_scalar(), facts::character_row()],
        diagnostics,
    )
}
