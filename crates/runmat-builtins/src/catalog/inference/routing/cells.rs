use crate::{BuiltinCatalogEntry, CellInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: CellInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        CellInferenceRule::Cellfun => {
            crate::catalog::entries::cells::cellfun::infer(request, entry)
        }
    }
}
