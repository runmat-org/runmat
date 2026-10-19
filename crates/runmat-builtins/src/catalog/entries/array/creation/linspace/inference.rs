use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::sequence_facts::infer(
        request,
        entry,
        super::super::sequence_facts::SequenceKind::Linear,
    )
}
