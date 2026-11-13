use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let (Some(LiteralValue::String(value)), Some(LiteralValue::String(set))) = (
        request.literals.literal_args.first(),
        request.literals.literal_args.get(1),
    ) {
        if value != set {
            reject(&mut diagnostics, 0, "literal text is not a member");
        }
    }

    // Unknown content never proves predicate success or failure. The public result
    // sequence is nevertheless exactly empty; shared framing handles output requests.
    finish(request, entry, vec![], diagnostics)
}
