use super::super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 0, Some(2), &mut diagnostics);
    for (index, value) in request.arguments.iter().enumerate() {
        if direct_integer(value) || resident_numeric(value) {
            diagnostics.push(argument_error(
                "RM-ROWFILTER-NUMERIC",
                "direct native integer and resident numeric arguments are not constructor roles",
                index,
            ));
        }
    }
    let variables = if request.arguments.is_empty() {
        empty_cell()
    } else {
        retained(request, 0)
    };
    let predicate = if request.arguments.len() < 2 {
        string()
    } else {
        retained(request, 1)
    };
    finish_fixed(
        entry,
        request,
        object(
            StaticClassIdentity::new("rowfilter"),
            &[("Variables", variables), ("Predicate", predicate)],
        ),
        diagnostics,
    )
}
