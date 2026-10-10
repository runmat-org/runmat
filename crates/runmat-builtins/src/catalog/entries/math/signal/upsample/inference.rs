use super::super::sample_rate_facts::{self, SampleDirection};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    sample_rate_facts::infer_sample(request, entry, SampleDirection::Up)
}
