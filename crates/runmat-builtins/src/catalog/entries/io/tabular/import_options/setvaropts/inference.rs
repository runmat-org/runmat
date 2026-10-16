use super::super::shared::{common_diagnostics, modified_options};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn setvaropts(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = common_diagnostics(request, "SETVAROPTS");
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SETVAROPTS-ARITY",
            "setvaropts requires an options object and at least one name/value pair",
            0,
        ));
    }
    finish_fixed(entry, request, modified_options(request), diagnostics)
}
