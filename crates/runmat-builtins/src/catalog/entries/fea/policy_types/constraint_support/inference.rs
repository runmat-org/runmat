use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn direction(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = exact_arity(request, 1, "fea.Direction.vector");
    if let Some(input) = request.arguments.first() {
        let supported = matches!(&input.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
            || matches!(input.kind, ValueKindFact::Logical | ValueKindFact::Unknown);
        if !supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-DIRECTION-COMPONENTS",
                "fea.Direction.vector requires real numeric components",
                0,
            ));
        }
        if input.shape.element_count().is_some_and(|count| count != 3) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-DIRECTION-SHAPE",
                "fea.Direction.vector requires exactly three components",
                0,
            ));
        }
    }
    finish_fixed(entry, request, object("fea.Direction"), diagnostics)
}

pub(super) fn matching(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object("fea.MatchingPeriodicPairing"),
        exact_arity(request, 0, "fea.PeriodicPairing.matching"),
    )
}

pub(super) fn geometric(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object("fea.GeometricPeriodicPairing"),
        exact_arity(request, 1, "fea.PeriodicPairing.geometric"),
    )
}

pub(super) fn identity(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object("fea.IdentityPeriodicTransform"),
        exact_arity(request, 0, "fea.PeriodicTransform.identity"),
    )
}

pub(super) fn translation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object("fea.TranslationPeriodicTransform"),
        exact_arity(request, 1, "fea.PeriodicTransform.translation"),
    )
}

pub(super) fn rotation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object("fea.RotationPeriodicTransform"),
        exact_arity(request, 3, "fea.PeriodicTransform.rotation"),
    )
}

fn exact_arity(
    request: &CallRequest,
    expected: usize,
    name: &'static str,
) -> Vec<InferenceDiagnostic> {
    if request.arguments.len() == expected {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-CATALOG-FEA-CONSTRAINT-SUPPORT-ARITY",
            format!("{name} requires exactly {expected} inputs"),
            request.arguments.len().min(expected),
        )]
    }
}

fn object(class: &'static str) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new(class).owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
