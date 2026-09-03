pub(super) mod acceleration;
pub(super) mod aggregate;
pub(super) mod array;
mod distributed;
pub(super) mod introspection;
mod local;
pub(super) mod math;
pub(super) mod parallel;
pub(super) mod stats;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    distributed::infer(entry, request)
}

pub(super) fn infer_local(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    local::infer(entry, request)
}
