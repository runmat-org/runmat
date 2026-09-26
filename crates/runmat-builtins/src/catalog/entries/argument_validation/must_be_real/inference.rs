use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(LiteralValue::Complex { imaginary, .. }) = request.literals.literal_args.first() {
        if imaginary.parse::<f64>().is_ok_and(|value| value != 0.0) {
            reject(&mut diagnostics, 0, "nonzero imaginary content is not real");
        }
    }

    // Unknown content never proves predicate success or failure. The public result
    // sequence is nevertheless exactly empty; shared framing handles output requests.
    finish(request, entry, vec![], diagnostics)
}
