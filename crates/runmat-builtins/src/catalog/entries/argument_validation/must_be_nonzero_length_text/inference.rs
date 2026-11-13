use super::super::inference::{definitely_non_text, finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if definitely_non_text(input, false) {
            reject(&mut diagnostics, 0, "expected supported text");
        }
        if matches!(request.literals.literal_args.first(), Some(LiteralValue::String(text) | LiteralValue::Character(text)) if text.is_empty())
        {
            reject(&mut diagnostics, 0, "expected nonzero-length text");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
