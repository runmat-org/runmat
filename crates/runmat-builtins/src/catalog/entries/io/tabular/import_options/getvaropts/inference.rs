use super::super::shared::{common_diagnostics, variable_options};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn getvaropts(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = common_diagnostics(request, "GETVAROPTS");
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GETVAROPTS-ARITY",
            "getvaropts requires options and selection",
            0,
        ));
    }
    finish_fixed(entry, request, variable_options(), diagnostics)
}
