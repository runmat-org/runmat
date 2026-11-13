use crate::{
    MirBody, MirExpansionSource, MirExpressionStep, MirLocalId, MirLocalKind, MirRvalue,
    MirStmtKind,
};
use runmat_builtins::BuiltinWorkspaceEffect;

/// Only an operation on the current workspace can change a source binding's
/// presence. Generic dynamic evaluation (notably evalin on another workspace)
/// is not evidence about this function's locals.
fn directly_exposes_current_workspace(value: &MirRvalue) -> bool {
    matches!(
        value,
        MirRvalue::Call(call)
            if matches!(
                call.workspace_effect,
                Some(BuiltinWorkspaceEffect::LoadsExternalBindings
                    | BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace)
            )
    )
}

/// Inspect the existing ordered expression-step carriers, including
/// contextual selectors and dotted-call argument preparation. A nested call
/// keeps its own typed builtin effect even when the enclosing rvalue is Index
/// or SubscriptChain and has no direct call effect of its own.
pub(super) fn exposes_current_workspace(value: &MirRvalue) -> bool {
    if directly_exposes_current_workspace(value) {
        return true;
    }
    if let MirRvalue::ShortCircuit { right_temps, .. } = value {
        return right_temps
            .iter()
            .any(|statement| statement_exposes_current_workspace(&statement.kind));
    }
    let mut found = false;
    value.visit_direct_expression_steps_dyn(&mut |steps| {
        found |= steps_expose_current_workspace(steps);
    });
    found
}

pub(super) fn expansion_exposes_current_workspace(source: &MirExpansionSource) -> bool {
    let mut found = false;
    source.visit_direct_expression_steps_dyn(&mut |steps| {
        found |= steps_expose_current_workspace(steps);
    });
    found
}

fn steps_expose_current_workspace(steps: &[MirExpressionStep]) -> bool {
    steps.iter().any(|step| match step {
        MirExpressionStep::Let { value, .. } => exposes_current_workspace(value),
        MirExpressionStep::CaptureSequence { source, .. } => {
            expansion_exposes_current_workspace(source)
        }
    })
}

pub(super) fn statement_exposes_current_workspace(statement: &MirStmtKind) -> bool {
    let outer = match statement {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => exposes_current_workspace(value),
        MirStmtKind::CaptureSequence { source, .. } => expansion_exposes_current_workspace(source),
        _ => false,
    };
    if outer {
        return true;
    }
    let mut nested = false;
    statement.visit_statement_expression_steps_dyn(&mut |steps| {
        nested |= steps_expose_current_workspace(steps);
    });
    nested
}

pub(super) fn current_scope_bindings(body: &MirBody) -> Vec<MirLocalId> {
    body.locals
        .iter()
        .filter(|local| {
            local.binding.is_some()
                && matches!(
                    local.kind,
                    MirLocalKind::Parameter
                        | MirLocalKind::Output
                        | MirLocalKind::Binding
                        | MirLocalKind::IncomingWorkspace
                        | MirLocalKind::ExternalWorkspace
                        | MirLocalKind::Capture
                )
        })
        .map(|local| local.id)
        .collect()
}
