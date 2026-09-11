use crate::catalog::inference::field_path::{is_selector, selector_facts};
use runmat_types::{
    infer_index, infer_index_mutation, infer_member_read, infer_member_write,
    AssignmentCreationPolicy, AssignmentShapePolicy, CallRequest, DynamicReason, IndexKind,
    IndexResultContext, MemberName, MutationContract, PlaceMutationKind, SequenceUse, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest) -> (ValueFact, Vec<runmat_types::InferenceDiagnostic>) {
    let Some(base) = request.arguments.first() else {
        return (ValueFact::unknown(DynamicReason::RuntimeValue), Vec::new());
    };
    if request.arguments.len() < 3 {
        return (base.clone(), Vec::new());
    }
    let rhs_index = request.arguments.len() - 1;
    let leading_selector = request.arguments.get(1).filter(|value| is_selector(value));
    let start = 1 + usize::from(leading_selector.is_some());
    if start >= rhs_index {
        return (base.clone(), Vec::new());
    }
    let mut diagnostics = Vec::new();
    let target = if let Some(selector) = leading_selector {
        let selected = infer_index(
            base,
            IndexKind::Paren,
            &selector_facts(Some(selector)),
            IndexResultContext::ReadSingle,
        );
        diagnostics.extend(selected.diagnostics);
        selected.fact
    } else {
        base.clone()
    };
    let updated = write_path(
        &target,
        start,
        rhs_index,
        &request.arguments[rhs_index],
        request,
        &mut diagnostics,
    );
    let output = if leading_selector.is_some() {
        erase_changed_schema(base)
    } else {
        updated
    };
    (output, diagnostics)
}

fn write_path(
    base: &ValueFact,
    position: usize,
    end: usize,
    rhs: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    let Some(name) = request.literals.literal_string_at(position) else {
        return erase_changed_schema(base);
    };
    let member = MemberName::from(name);
    let next = position + 1;
    let has_selector = next < end && request.arguments.get(next).is_some_and(is_selector);
    let child_position = next + usize::from(has_selector);
    let assigned = if !has_selector && child_position >= end {
        rhs.clone()
    } else {
        let read = infer_member_read(base, &member, SequenceUse::RequireSingle);
        diagnostics.extend(read.diagnostics);
        if has_selector {
            write_indexed(
                &read.fact,
                next,
                child_position,
                end,
                rhs,
                request,
                diagnostics,
            )
        } else {
            write_path(&read.fact, child_position, end, rhs, request, diagnostics)
        }
    };
    let written = infer_member_write(base, &member, &assigned, true);
    diagnostics.extend(written.diagnostics);
    written.fact
}

fn write_indexed(
    field: &ValueFact,
    selector_position: usize,
    child_position: usize,
    end: usize,
    rhs: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    let selector = selector_facts(request.arguments.get(selector_position));
    if child_position >= end {
        let mutation = infer_index_mutation(
            field,
            &selector,
            rhs,
            indexed_assignment_contract(AssignmentCreationPolicy::CreateArrayByIndex),
        );
        diagnostics.extend(mutation.diagnostics);
        return mutation.fact;
    }
    let selected = infer_index(
        field,
        IndexKind::Paren,
        &selector,
        IndexResultContext::ReadSingle,
    );
    diagnostics.extend(selected.diagnostics);
    let updated = write_path(
        &selected.fact,
        child_position,
        end,
        rhs,
        request,
        diagnostics,
    );
    let mutation = infer_index_mutation(
        field,
        &selector,
        &updated,
        indexed_assignment_contract(AssignmentCreationPolicy::ExistingOnly),
    );
    diagnostics.extend(mutation.diagnostics);
    mutation.fact
}

fn indexed_assignment_contract(creation: AssignmentCreationPolicy) -> MutationContract {
    MutationContract {
        kind: PlaceMutationKind::IndexedAssign,
        creation,
        shape: AssignmentShapePolicy::MatlabCompatible,
    }
}

fn erase_changed_schema(base: &ValueFact) -> ValueFact {
    let mut output = base.clone();
    match &mut output.kind {
        ValueKindFact::Struct(structure) => {
            structure.fields_complete = false;
            structure.elements.clear();
            structure.elements_complete = false;
        }
        ValueKindFact::Object(object) => object.properties_complete = false,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    }
    output
}
