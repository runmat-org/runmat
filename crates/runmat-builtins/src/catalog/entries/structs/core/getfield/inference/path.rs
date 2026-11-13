use crate::catalog::inference::field_path::{is_selector, selector_facts};
use runmat_types::{
    infer_index, infer_member_read, CallRequest, DynamicReason, IndexKind, IndexResultContext,
    MemberName, SequenceUse, ValueFact,
};

pub(super) fn infer(request: &CallRequest) -> (ValueFact, Vec<runmat_types::InferenceDiagnostic>) {
    let Some(mut current) = request.arguments.first().cloned() else {
        return (ValueFact::unknown(DynamicReason::RuntimeValue), Vec::new());
    };
    let mut diagnostics = Vec::new();
    let mut position = 1;
    if request.arguments.get(position).is_some_and(is_selector) {
        let inferred = infer_index(
            &current,
            IndexKind::Paren,
            &selector_facts(request.arguments.get(position)),
            IndexResultContext::ReadSingle,
        );
        current = inferred.fact;
        diagnostics.extend(inferred.diagnostics);
        position += 1;
    }
    while position < request.arguments.len() {
        let Some(name) = request.literals.literal_string_at(position) else {
            current = ValueFact::unknown(DynamicReason::RuntimeValue);
            position += 1;
            if request.arguments.get(position).is_some_and(is_selector) {
                position += 1;
            }
            continue;
        };
        let inferred = infer_member_read(
            &current,
            &MemberName::from(name),
            SequenceUse::RequireSingle,
        );
        current = inferred.fact;
        diagnostics.extend(inferred.diagnostics);
        position += 1;
        if request.arguments.get(position).is_some_and(is_selector) {
            let inferred = infer_index(
                &current,
                IndexKind::Paren,
                &selector_facts(request.arguments.get(position)),
                IndexResultContext::ReadSingle,
            );
            current = inferred.fact;
            diagnostics.extend(inferred.diagnostics);
            position += 1;
        }
    }
    (current, diagnostics)
}
