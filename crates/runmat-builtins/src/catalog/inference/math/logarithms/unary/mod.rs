mod output;
mod policy;

use super::super::super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::super::super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, LogarithmKind};
use runmat_types::{CallInference, CallRequest, DynamicReason, StorageFact, ValueFact};

pub(super) fn infer(
    kind: LogarithmKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let (output, diagnostics) = infer_value(kind, request);
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_value(
    kind: LogarithmKind,
    request: &CallRequest,
) -> (ValueFact, Vec<runmat_types::InferenceDiagnostic>) {
    let policy = policy::UnaryLogarithmPolicy::for_kind(kind);
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            policy.arity_diagnostic(),
            format!("{} requires exactly one input", policy.name()),
            0,
        ));
        return (ValueFact::unknown(DynamicReason::RuntimeValue), diagnostics);
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            policy.arity_diagnostic(),
            format!("{} accepts exactly one input", policy.name()),
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            policy.sparse_diagnostic(),
            format!("{} does not currently accept sparse input", policy.name()),
            0,
        ));
        return (
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(|literal| super::literal_domain::infer(literal, policy.real_boundary()));
    let mut inferred = output::infer(policy, input, literal_domain, &mut diagnostics);
    if inferred.materialize {
        materialize(&mut inferred.value);
    } else if inferred.preserve_dynamic_shape {
        preserve_shape_as_dynamic(&mut inferred.value);
    }
    (inferred.value, diagnostics)
}
