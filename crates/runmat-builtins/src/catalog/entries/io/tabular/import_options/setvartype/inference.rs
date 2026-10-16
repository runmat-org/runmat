use super::super::shared::{common_diagnostics, modified_options};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn setvartype(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = common_diagnostics(request, "SETVARTYPE");
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SETVARTYPE-ARITY",
            "setvartype requires options, selection, and type",
            0,
        ));
    }
    finish_fixed(entry, request, modified_options(request), diagnostics)
}
