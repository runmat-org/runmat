use crate::{
    AsyncBehaviorFact, MirBody, MirLocal, MirLocalId, MirLocalKind, MirOperand, MirPlace,
    MirRvalue, MirStmtKind, MirTerminatorKind,
};
use runmat_hir::{BindingId, FunctionId, RequestedOutputCount, SpawnSafetyFact, TypeFact};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{
    AccelEligibilityFact, AnalysisStore, EffectSummary, FusibilityFact, ParallelSafetyFact,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionSummary {
    pub function: FunctionId,
    pub outputs: Vec<TypeFact>,
    pub requested_output_sensitive: Vec<(RequestedOutputCount, Vec<TypeFact>)>,
    pub effects: EffectSummary,
    pub reads_captures: BTreeSet<BindingId>,
    pub writes_captures: BTreeSet<BindingId>,
    pub spawn_safety: SpawnSafetyFact,
    pub fusibility: FusibilityFact,
    pub parallel_safety: ParallelSafetyFact,
    pub accel_eligibility: AccelEligibilityFact,
}

pub fn summarize_body(body: &MirBody, store: &mut AnalysisStore) -> FunctionSummary {
    let mut output_count = 0;
    let mut async_behavior = AsyncBehaviorFact::NeverSuspends;
    let mut workspace = Vec::new();
    let mut reads_captures = BTreeSet::new();
    let mut writes_captures = BTreeSet::new();

    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { place, value } => {
                    scan_rvalue(body, value, &mut reads_captures, &mut async_behavior);
                    scan_place_write(body, place, &mut writes_captures);
                }
                MirStmtKind::MultiAssign { targets, value } => {
                    scan_rvalue(body, value, &mut reads_captures, &mut async_behavior);
                    for target in &targets.targets {
                        if let crate::MirOutputTarget::Place(place) = target {
                            scan_place_write(body, place, &mut writes_captures);
                        }
                    }
                }
                MirStmtKind::Expr(value) => {
                    scan_rvalue(body, value, &mut reads_captures, &mut async_behavior);
                }
                MirStmtKind::PlaceMutation(_) => {}
                MirStmtKind::WorkspaceEffect { effect, .. } => workspace.push(effect.clone()),
            }
        }

        match &block.terminator.kind {
            MirTerminatorKind::Branch { cond, .. } => {
                scan_operand(body, cond, &mut reads_captures);
            }
            MirTerminatorKind::For {
                binding, iterable, ..
            } => {
                scan_rvalue(body, iterable, &mut reads_captures, &mut async_behavior);
                scan_local_write(body, *binding, &mut writes_captures);
            }
            MirTerminatorKind::Return(outputs) => {
                output_count = output_count.max(outputs.len());
                for output in outputs {
                    scan_operand(body, output, &mut reads_captures);
                }
            }
            MirTerminatorKind::Await { future, .. } => {
                async_behavior = AsyncBehaviorFact::RequiresAsyncRuntime;
                scan_operand(body, future, &mut reads_captures);
            }
            MirTerminatorKind::Goto(_)
            | MirTerminatorKind::TryCatch { .. }
            | MirTerminatorKind::Unreachable => {}
        }
    }

    let summary = FunctionSummary {
        function: body.function,
        outputs: vec![TypeFact::Unknown; output_count],
        requested_output_sensitive: Vec::new(),
        effects: EffectSummary {
            workspace,
            environment: Vec::new(),
            async_behavior: Some(async_behavior),
        },
        reads_captures,
        writes_captures,
        spawn_safety: SpawnSafetyFact::RequiresIsolation,
        fusibility: FusibilityFact::Unknown,
        parallel_safety: ParallelSafetyFact::Unknown,
        accel_eligibility: AccelEligibilityFact::Unknown,
    };
    store.functions.insert(body.function, summary.clone());
    summary
}

fn scan_rvalue(
    body: &MirBody,
    value: &MirRvalue,
    reads_captures: &mut BTreeSet<BindingId>,
    async_behavior: &mut AsyncBehaviorFact,
) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) => {
            scan_operand(body, operand, reads_captures);
        }
        MirRvalue::Binary(left, _, right) => {
            scan_operand(body, left, reads_captures);
            scan_operand(body, right, reads_captures);
        }
        MirRvalue::Range { start, step, end } => {
            scan_operand(body, start, reads_captures);
            if let Some(step) = step {
                scan_operand(body, step, reads_captures);
            }
            scan_operand(body, end, reads_captures);
        }
        MirRvalue::Call(call) => {
            for arg in &call.args {
                scan_operand(body, arg, reads_captures);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                scan_operand(body, element, reads_captures);
            }
        }
        MirRvalue::Index { base, .. } => scan_operand(body, base, reads_captures),
        MirRvalue::Future(_) => {}
        MirRvalue::Spawn(future) => {
            *async_behavior = AsyncBehaviorFact::RequiresAsyncRuntime;
            scan_operand(body, future, reads_captures);
        }
    }
}

fn scan_place_write(body: &MirBody, place: &MirPlace, writes_captures: &mut BTreeSet<BindingId>) {
    if let Some(local) = place_root(place) {
        scan_local_write(body, local, writes_captures);
    }
}

fn place_root(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::Index(base, _) => place_root(base),
        MirPlace::Binding(_) => None,
    }
}

fn scan_operand(body: &MirBody, operand: &MirOperand, reads_captures: &mut BTreeSet<BindingId>) {
    if let MirOperand::Local(local) = operand {
        if let Some(binding) = capture_binding(body, *local) {
            reads_captures.insert(binding);
        }
    }
}

fn scan_local_write(body: &MirBody, local: MirLocalId, writes_captures: &mut BTreeSet<BindingId>) {
    if let Some(binding) = capture_binding(body, local) {
        writes_captures.insert(binding);
    }
}

fn capture_binding(body: &MirBody, local: MirLocalId) -> Option<BindingId> {
    local_for_id(body, local).and_then(|local| {
        if matches!(local.kind, MirLocalKind::Capture) {
            local.binding
        } else {
            None
        }
    })
}

fn local_for_id(body: &MirBody, local: MirLocalId) -> Option<&MirLocal> {
    body.locals.iter().find(|candidate| candidate.id == local)
}
