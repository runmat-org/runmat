use super::GRIDDED_INTERPOLANT_CLASS_IDENTITY;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ObjectFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let properties = ["GridVectors", "Values", "Method", "ExtrapolationMethod"]
        .into_iter()
        .map(|name| {
            (
                name.to_owned(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(GRIDDED_INTERPOLANT_CLASS_IDENTITY.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish(request, entry, output, Vec::new())
}

pub(super) fn subsref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GRIDDED-SUBSREF-ARITY",
            "griddedInterpolant.subsref requires an interpolant and an indexing path",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if let ValueKindFact::Object(object) = &receiver.kind {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|identity| identity != &GRIDDED_INTERPOLANT_CLASS_IDENTITY.owned())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GRIDDED-SUBSREF-RECEIVER",
                    "expected a griddedInterpolant receiver",
                    0,
                ));
            }
        }
    }
    // The standard substruct path can read a property or evaluate numeric
    // queries. Without decoding that runtime path, neither kind nor shape is
    // soundly known.
    finish(
        request,
        entry,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: ValueFact,
    diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
