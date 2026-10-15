use crate::catalog::inference::{default_double_scalar, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, InferenceDiagnostic, NumericClass, NumericDomain, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
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
                || numeric.domain == NumericDomain::Complex
                || argument
                    .shape
                    .element_count()
                    .is_some_and(|count| count != 1)
                || matches!(argument.storage, StorageFact::Sparse | StorageFact::Opaque)
                || matches!(
                    argument.residency,
                    ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                )
            {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM-CATALOG-TOC-TOKEN",
                    "toc token must be an opaque uint64 scalar",
                ));
            }
        }
    }
    let mut result = finish_fixed(entry, request, default_double_scalar(), diagnostics);
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0)
    {
        result
            .effects
            .0
            .remove(&runmat_types::EffectKind::UserInterface);
        result
            .effects
            .0
            .remove(&runmat_types::EffectKind::HostCallback);
        result
            .capabilities
            .0
            .remove(&runmat_types::CapabilityRequirement::UserInterface);
    }
    result
}
