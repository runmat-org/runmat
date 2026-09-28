use crate::{
    catalog::entries::io::text_admission::{finish, input, reject, Role},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 || !request.arguments.len().saturating_sub(2).is_multiple_of(2) {
        reject(
            &mut diagnostics,
            0,
            "expected required inputs and paired supported text options",
        );
    }
    input(request, 0, Role::Lines, &mut diagnostics);
    input(request, 1, Role::Scalar, &mut diagnostics);
    for index in (2..request.arguments.len()).step_by(2) {
        input(request, index, Role::Scalar, &mut diagnostics);
        input(request, index + 1, Role::Scalar, &mut diagnostics);
        if let Some(option) = request.literals.literal_string_at(index) {
            if option != "lineending" {
                reject(&mut diagnostics, index, "unsupported option");
            } else if request
                .literals
                .literal_string_at(index + 1)
                .is_some_and(|value| {
                    !matches!(
                        value.as_str(),
                        "unix" | "lf" | "windows" | "pc" | "crlf" | "mac" | "cr"
                    )
                })
            {
                reject(&mut diagnostics, index + 1, "unsupported option value");
            }
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish(request, entry, output, diagnostics)
}
