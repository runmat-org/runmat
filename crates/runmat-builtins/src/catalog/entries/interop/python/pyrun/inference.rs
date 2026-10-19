use super::super::inference_support::{run, RunSource};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    run(request, entry, RunSource::Code)
}
