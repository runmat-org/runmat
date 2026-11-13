use super::super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 2, Some(3), &mut diagnostics);
    if request.arguments.len() > 2 {
        text(request, 2, &mut diagnostics);
        if request
            .literals
            .literal_string_at(2)
            .is_some_and(|text| !["openright", "openleft", "open", "closed"].contains(&text.trim()))
        {
            diagnostics.push(argument_error(
                "RM-TIMERANGE-INCLUSIVITY",
                "unsupported interval inclusivity",
                2,
            ));
        }
    }
    // Bounds are retained verbatim. Constructor validation does not prove a
    // scalar datetime/duration or compatibility admission for unknown values.
    finish_fixed(
        entry,
        request,
        object(
            StaticClassIdentity::new("timerange"),
            &[
                ("Start", retained(request, 0)),
                ("End", retained(request, 1)),
                ("Inclusivity", string()),
            ],
        ),
        diagnostics,
    )
}
