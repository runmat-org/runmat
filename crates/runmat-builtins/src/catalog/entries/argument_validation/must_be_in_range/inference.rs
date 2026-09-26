use super::super::inference::{definitely_non_text, finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    for index in 3..request.arguments.len() {
        if request
            .arguments
            .get(index)
            .is_some_and(|value| definitely_non_text(value, false))
        {
            reject(&mut diagnostics, index, "expected text endpoint option");
        }
        if let Some(flag) = request.literals.literal_string_at(index) {
            let allowed = if request.arguments.len() == 4 {
                matches!(
                    flag.trim(),
                    "inclusive"
                        | "exclusive"
                        | "exclude-lower"
                        | "openleft"
                        | "open-left"
                        | "exclude-upper"
                        | "openright"
                        | "open-right"
                )
            } else {
                matches!(flag.trim(), "inclusive" | "exclusive")
            };
            if !allowed {
                reject(&mut diagnostics, index, "invalid range endpoint option");
            }
        }
    }
    if let [value, lower, upper, ..] = request.arguments.as_slice() {
        for bound in [lower, upper] {
            if let (ValueKindFact::Numeric(a), ValueKindFact::Numeric(b)) =
                (&value.kind, &bound.kind)
            {
                if a.class != b.class {
                    reject(
                        &mut diagnostics,
                        1,
                        "range bounds must share the value class",
                    );
                }
            }
        }
    }

    // Unknown content never proves predicate success or failure. The public result
    // sequence is nevertheless exactly empty; shared framing handles output requests.
    finish(request, entry, vec![], diagnostics)
}
