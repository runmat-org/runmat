use crate::catalog::inference::{default_double_scalar, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, InferenceDiagnostic, NumericClass, NumericDomain, ValueFact,
    ValueKindFact,
};

pub(super) fn tic(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![InferenceDiagnostic::error(
            "RM-CATALOG-TIC-ARITY",
            "tic takes no inputs",
        )]
    };
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(numeric_kind(NumericClass::UInt64, NumericDomain::Real)),
        diagnostics,
    )
}

pub(super) fn toc(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-TOC-ARITY",
            "toc accepts at most one timer token",
        ));
    } else if let Some(argument) = request.arguments.first() {
        if !matches!(
            argument.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM-CATALOG-TOC-TOKEN",
                "toc token must be an opaque uint64 scalar",
            ));
        } else if let ValueKindFact::Numeric(ref numeric) = argument.kind {
            if numeric.class != NumericClass::UInt64
                || argument
                    .shape
                    .element_count()
                    .is_some_and(|count| count != 1)
            {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM-CATALOG-TOC-TOKEN",
                    "toc token must be an opaque uint64 scalar",
                ));
            }
        }
    }
    finish_fixed(entry, request, default_double_scalar(), diagnostics)
}
