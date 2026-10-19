use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if match input.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Numeric(numeric) => {
                !matches!(numeric.class, NumericClass::Double | NumericClass::Single)
            }
            _ => true,
        } {
            reject(&mut diagnostics, 0, "expected floating-point numeric data");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
