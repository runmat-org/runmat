//! Shared abstract call framing, not identity dispatch or a runtime option parser.
use crate::catalog::inference::argument_error;
use crate::{BuiltinCatalogEntry, BuiltinParamArity};
use runmat_types::*;

/// Only these facts describe stored array dimensions inspected by validators.
/// Execution/foreign handles do not imply partition or foreign-payload access.
pub(super) fn has_stored_shape(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Symbolic
            | ValueKindFact::Cell(_)
            | ValueKindFact::Struct(_)
            | ValueKindFact::Object(_)
    )
}

pub(super) fn reject(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
    message: &'static str,
) {
    diagnostics.push(argument_error(
        "RM-ARGUMENT-VALIDATION-INPUT",
        message,
        index,
    ));
}

pub(super) fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let count = request.arguments.len();
    let admitted = entry.descriptor.signatures.iter().any(|form| {
        let minimum = form
            .inputs
            .iter()
            .filter(|p| p.arity == BuiltinParamArity::Required)
            .count();
        let variadic = form
            .inputs
            .iter()
            .any(|p| p.arity == BuiltinParamArity::Variadic);
        count >= minimum && (variadic || count <= form.inputs.len())
    });
    if !admitted {
        diagnostics.push(argument_error(
            "RM-ARGUMENT-VALIDATION-ARITY",
            "unsupported number of validator inputs",
            0,
        ));
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

/// True only for facts proving rejection by the shared text admission.
/// Cell contents are never inferred from container shape alone.
pub(super) fn definitely_non_text(value: &ValueFact, scalar: bool) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => scalar && value.shape.element_count().is_some_and(|n| n != 1),
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|extent| extent.is_some_and(|n| n != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|extent| extent.is_some_and(|n| n != 1))
        }),
        ValueKindFact::Cell(cell) if !scalar => {
            cell.elements_complete
                && cell
                    .elements
                    .iter()
                    .any(|value| definitely_non_text(value, false))
        }
        _ => true,
    }
}
