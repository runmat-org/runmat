//! Abstract roles for file-text calls, not a runtime option parser.
//! Unlike search-path mutation, these calls reject numeric text conversions and
//! nested cells. Only direct scalar text elements are accepted in a text cell.
use super::repl_fs::text_input;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum Role {
    Scalar,
    Paths,
    Lines,
}

pub(super) fn supports(value: &ValueFact, role: Role) -> bool {
    match (&value.kind, role) {
        (_, Role::Scalar) => text_input::scalar(value),
        (ValueKindFact::Unknown | ValueKindFact::String, _) => true,
        (ValueKindFact::Character, Role::Paths) => text_input::scalar(value),
        (ValueKindFact::Character, Role::Lines) => text_input::character_container(&value.shape),
        (ValueKindFact::Cell(cell), _) => {
            if cell.elements_complete {
                cell.elements.iter().all(text_input::scalar)
            } else {
                text_input::scalar(&cell.element)
            }
        }
        _ => false,
    }
}

pub(super) fn reject(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
    message: &'static str,
) {
    diagnostics.push(argument_error("RM-TEXT-IO-INPUT", message, index));
}

pub(super) fn input(
    request: &CallRequest,
    index: usize,
    role: Role,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request
        .arguments
        .get(index)
        .is_some_and(|value| !supports(value, role))
    {
        reject(diagnostics, index, "unsupported host text representation");
    }
}

pub(super) fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: ValueFact,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
