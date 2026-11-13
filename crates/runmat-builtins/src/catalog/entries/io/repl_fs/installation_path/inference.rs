use crate::{BuiltinCatalogEntry, InstallationPathInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs) fn infer(
    rule: InstallationPathInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        InstallationPathInferenceRule::Root => super::matlabroot::inference::infer(request, entry),
    }
}
