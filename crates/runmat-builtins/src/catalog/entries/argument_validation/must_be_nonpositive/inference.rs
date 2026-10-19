use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(LiteralValue::Number(value)) = request.literals.literal_args.first() {
        if !matches!(
            value.partial_cmp(&0.0),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            reject(&mut diagnostics, 0, "literal value fails the validator");
        }
    }

    // Unknown content never proves predicate success or failure. The public result
    // sequence is nevertheless exactly empty; shared framing handles output requests.
    finish(request, entry, vec![], diagnostics)
}
