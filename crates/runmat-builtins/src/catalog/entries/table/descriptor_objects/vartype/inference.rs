use super::super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, Some(1), &mut diagnostics);
    text(request, 0, &mut diagnostics);
    if request.arguments.first().is_some_and(resident_numeric) {
        diagnostics.push(argument_error(
            "RM-VARTYPE-RESIDENT",
            "type must be host text; numeric device arguments reject before gather",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        object(StaticClassIdentity::new("vartype"), &[("Type", string())]),
        diagnostics,
    )
}
