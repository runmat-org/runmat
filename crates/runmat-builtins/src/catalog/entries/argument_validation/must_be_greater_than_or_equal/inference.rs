use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let (Some(LiteralValue::Number(lhs)), Some(LiteralValue::Number(rhs))) = (
        request.literals.literal_args.first(),
        request.literals.literal_args.get(1),
    ) {
        if !matches!(
            lhs.partial_cmp(rhs),
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        ) {
            reject(
                &mut diagnostics,
                0,
                "literal comparison fails the validator",
            );
        }
    }

    // Unknown content never proves predicate success or failure. The public result
    // sequence is nevertheless exactly empty; shared framing handles output requests.
    finish(request, entry, vec![], diagnostics)
}
