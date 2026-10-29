//! Structural recognition only. A selected stochastic route must separately
//! prove numeric classes, shape, output publication, and ordered RNG effects.

use runmat_hir::{CallSyntax, CallableFallbackPolicy, CallableIdentity, OperatorKind};
use runmat_mir::{
    BasicBlockId, MirBody, MirCall, MirCallArg, MirCallee, MirConstant, MirLocalId, MirLocalKind,
    MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
};

mod snapshots;

pub(super) struct MirStochasticEvolutionPlan {
    pub(super) iteration_binding: MirLocalId,
    pub(super) noise: MirLocalId,
    pub(super) state: MirLocalId,
    pub(super) drift: MirOperand,
    pub(super) scale: MirOperand,
    pub(super) steps: MirOperand,
    pub(super) randn_args: Vec<MirOperand>,
}

fn disabled() -> bool {
    std::env::var("RUNMAT_DISABLE_STOCHASTIC_EVOLUTION")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

fn is_one(operand: &MirOperand) -> bool {
    match operand {
        MirOperand::Constant(MirConstant::Number(value)) => value.parse::<f64>().ok() == Some(1.0),
        MirOperand::Constant(MirConstant::IntegerLiteral(value)) => {
            runmat_value::IntValue::from(value).try_to_u64() == Some(1)
        }
        _ => false,
    }
}

fn single_output_builtin(call: &MirCall, name: &str) -> bool {
    let Some(expected) = runmat_builtins::builtin_id_by_name(name) else {
        return false;
    };
    matches!(&call.callee, MirCallee::Static(CallableIdentity::Builtin(actual)) if *actual == expected)
        && call.syntax == CallSyntax::Plain
        && call.fallback_policy == CallableFallbackPolicy::None
        && call.requested_outputs.known_count() == Some(1)
        && call.workspace_first_name.is_none()
        && !call.bare_identifier
}

fn assigned_local(stmt: &MirStmt) -> Option<(MirLocalId, &MirRvalue)> {
    let MirStmtKind::Assign {
        place: MirPlace::Local(local),
        value,
    } = &stmt.kind
    else {
        return None;
    };
    Some((*local, value))
}

fn invariant(operand: &MirOperand, written: &[MirLocalId], binding: MirLocalId) -> bool {
    match operand {
        MirOperand::Local(id) => *id != binding && !written.contains(id),
        MirOperand::Constant(_) => true,
        MirOperand::FunctionHandle(_) => false,
    }
}

fn source_binding(body: &MirBody, id: MirLocalId) -> bool {
    body.locals
        .iter()
        .find(|local| local.id == id)
        .is_some_and(|local| match &local.kind {
            MirLocalKind::Parameter
            | MirLocalKind::Output
            | MirLocalKind::Binding
            | MirLocalKind::IncomingWorkspace => true,
            // An external workspace local is a source operand, but unlike an
            // incoming workspace local it is resolved dynamically by the
            // executor. Structural recognition must not recast it as an
            // initialized parameter/local; the runtime lookup remains the
            // authority for whether the binding exists.
            MirLocalKind::ExternalWorkspace => true,
            MirLocalKind::Temporary | MirLocalKind::Capture => false,
        })
}

/// Recognize the entire, ordered source body. Any extra statement or alternate
/// call target stays on the original For path, including its ordinary effects.
pub(super) fn detect(
    body: &MirBody,
    header_block: BasicBlockId,
    binding: MirLocalId,
    body_block: BasicBlockId,
    iterable: &MirRvalue,
) -> Option<MirStochasticEvolutionPlan> {
    if disabled() {
        return None;
    }
    let MirRvalue::Range { start, step, end } = iterable else {
        return None;
    };
    if !is_one(start) || !step.as_ref().is_none_or(is_one) {
        return None;
    }
    let block = body.blocks.iter().find(|block| block.id == body_block)?;
    if !matches!(&block.terminator.kind, MirTerminatorKind::Goto(target) if *target == header_block)
    {
        return None;
    }
    let mut source = snapshots::SourceBody::parse(body, &block.statements)?;
    let [noise_stmt, scaled_stmt, argument_stmt, exponent_stmt, state_stmt] = source.statements;
    let assignments = [
        assigned_local(noise_stmt)?,
        assigned_local(scaled_stmt)?,
        assigned_local(argument_stmt)?,
        assigned_local(exponent_stmt)?,
        assigned_local(state_stmt)?,
    ];
    let mut written = [binding; 7];
    let mut written_count = 0;
    for id in assignments
        .iter()
        .map(|(id, _)| *id)
        .chain(source.snapshot_targets())
    {
        written[written_count] = id;
        written_count += 1;
    }
    let written = &written[..written_count];
    if written.contains(&binding)
        || written
            .iter()
            .enumerate()
            .any(|(index, id)| written[..index].contains(id))
        || !invariant(end, written, binding)
    {
        return None;
    }

    let [(noise, MirRvalue::Call(randn)), (
        scaled,
        MirRvalue::Binary(scale, OperatorKind::ElementwiseMultiply, MirOperand::Local(noise_use)),
    ), (argument, MirRvalue::Binary(drift, OperatorKind::Add, MirOperand::Local(scaled_use))), (exponent, MirRvalue::Call(exp)), (
        state,
        MirRvalue::Binary(
            MirOperand::Local(state_use),
            OperatorKind::ElementwiseMultiply,
            MirOperand::Local(exponent_use),
        ),
    )] = &assignments
    else {
        return None;
    };
    let state_source = source.state_source(*state_use)?;
    let drift = source.drift_source(drift)?;
    if !source.all_consumed()
        || noise != noise_use
        || scaled != scaled_use
        || *state != state_source
        || exponent != exponent_use
    {
        return None;
    }
    // Only compiler temporaries may disappear with the fused arithmetic.
    // Identical-looking source assignments to named intermediates remain
    // observable through the workspace and must execute on the original path.
    if [scaled, argument, exponent].into_iter().any(|id| {
        !body
            .locals
            .iter()
            .any(|local| local.id == *id && matches!(local.kind, MirLocalKind::Temporary))
    }) {
        return None;
    }
    if !single_output_builtin(randn, "randn")
        || !randn.args.iter().all(
            |arg| matches!(arg, MirCallArg::Single(value) if invariant(value, written, binding)),
        )
        || !single_output_builtin(exp, "exp")
        || !matches!(exp.args.as_slice(), [MirCallArg::Single(MirOperand::Local(id))] if id == argument)
        || !invariant(scale, written, binding)
        || !invariant(&drift, written, binding)
        || !source_binding(body, *noise)
        || !source_binding(body, *state)
    {
        return None;
    }
    let randn_args = randn
        .args
        .iter()
        .map(|arg| match arg {
            MirCallArg::Single(value) => Some(value.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(MirStochasticEvolutionPlan {
        iteration_binding: binding,
        noise: *noise,
        state: *state,
        drift,
        scale: scale.clone(),
        steps: end.clone(),
        randn_args,
    })
}
