use crate::{
    AsyncBehaviorFact, MirBody, MirCall, MirCallArg, MirIndexComponent, MirIndexing, MirLocal,
    MirLocalId, MirLocalKind, MirOperand, MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
};
use runmat_hir::{
    BindingId, FunctionId, HirCallableRef, PlaceMutation, RequestedOutputCount, SpawnSafetyFact,
    TypeFact, WorkspaceEffect,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{
    AccelEligibilityFact, AnalysisStore, EffectSummary, FusibilityFact, ParallelSafetyFact,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionSummary {
    pub function: FunctionId,
    pub abi: FunctionAbiSummary,
    pub outputs: Vec<TypeFact>,
    pub requested_output_sensitive: Vec<(RequestedOutputCount, Vec<TypeFact>)>,
    pub effects: EffectSummary,
    pub reads_captures: BTreeSet<BindingId>,
    pub writes_captures: BTreeSet<BindingId>,
    pub writes_globals: BTreeSet<BindingId>,
    pub writes_persistents: BTreeSet<BindingId>,
    pub may_call_unknown: bool,
    pub place_mutations: Vec<PlaceMutation>,
    pub calls: Vec<CallSummary>,
    pub spawn_safety: SpawnSafetyFact,
    pub fusibility: FusibilityFact,
    pub parallel_safety: ParallelSafetyFact,
    pub accel_eligibility: AccelEligibilityFact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallSummary {
    pub callee: HirCallableRef,
    pub requested_outputs: RequestedOutputCount,
    pub arg_count: usize,
    pub expansion_arg_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionAbiSummary {
    pub fixed_inputs: Vec<BindingId>,
    pub varargin: Option<BindingId>,
    pub fixed_outputs: Vec<BindingId>,
    pub varargout: Option<BindingId>,
    pub implicit_nargin: Option<BindingId>,
    pub implicit_nargout: Option<BindingId>,
}

pub fn summarize_body(body: &MirBody, store: &mut AnalysisStore) -> FunctionSummary {
    let mut output_count = 0;
    let mut async_behavior = AsyncBehaviorFact::NeverSuspends;
    let mut workspace = Vec::new();
    let mut environment = Vec::new();
    let mut reads_captures = BTreeSet::new();
    let mut writes_captures = BTreeSet::new();
    let mut writes_globals = BTreeSet::new();
    let mut writes_persistents = BTreeSet::new();
    let mut calls = Vec::new();
    let mut may_call_unknown = false;
    let mut place_mutations = Vec::new();

    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { place, value } => {
                    scan_rvalue(
                        body,
                        value,
                        &mut reads_captures,
                        &mut async_behavior,
                        &mut calls,
                        &mut may_call_unknown,
                    );
                    scan_place_write(body, place, &mut writes_captures);
                }
                MirStmtKind::MultiAssign { targets, value } => {
                    scan_rvalue(
                        body,
                        value,
                        &mut reads_captures,
                        &mut async_behavior,
                        &mut calls,
                        &mut may_call_unknown,
                    );
                    for target in &targets.targets {
                        if let crate::MirOutputTarget::Place(place) = target {
                            scan_place_write(body, place, &mut writes_captures);
                        }
                    }
                }
                MirStmtKind::Expr(value) => {
                    scan_rvalue(
                        body,
                        value,
                        &mut reads_captures,
                        &mut async_behavior,
                        &mut calls,
                        &mut may_call_unknown,
                    );
                }
                MirStmtKind::PlaceMutation(mutation) => place_mutations.push(mutation.clone()),
                MirStmtKind::WorkspaceEffect { effect, bindings } => {
                    scan_workspace_effect(
                        body,
                        effect,
                        bindings,
                        &mut writes_globals,
                        &mut writes_persistents,
                    );
                    workspace.push(effect.clone());
                }
                MirStmtKind::EnvironmentEffect(effect) => environment.push(effect.clone()),
            }
        }

        match &block.terminator.kind {
            MirTerminatorKind::Branch { cond, .. } => {
                scan_operand(body, cond, &mut reads_captures);
            }
            MirTerminatorKind::Switch { discr, cases, .. } => {
                scan_operand(body, discr, &mut reads_captures);
                for (case, _) in cases {
                    scan_operand(body, case, &mut reads_captures);
                }
            }
            MirTerminatorKind::For {
                binding, iterable, ..
            } => {
                scan_rvalue(
                    body,
                    iterable,
                    &mut reads_captures,
                    &mut async_behavior,
                    &mut calls,
                    &mut may_call_unknown,
                );
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

    let fusibility = classify_fusibility(&workspace, &environment, may_call_unknown);
    let parallel_safety = classify_parallel_safety(&workspace, &environment, may_call_unknown);
    let accel_eligibility = classify_accel_eligibility(&workspace, &environment, may_call_unknown);

    let summary = FunctionSummary {
        function: body.function,
        abi: FunctionAbiSummary {
            fixed_inputs: body.abi.fixed_inputs.clone(),
            varargin: body.abi.varargin,
            fixed_outputs: body.abi.fixed_outputs.clone(),
            varargout: body.abi.varargout,
            implicit_nargin: body.abi.implicit_nargin,
            implicit_nargout: body.abi.implicit_nargout,
        },
        outputs: vec![TypeFact::Unknown; output_count],
        requested_output_sensitive: Vec::new(),
        effects: EffectSummary {
            workspace,
            environment,
            async_behavior: Some(async_behavior),
        },
        reads_captures,
        writes_captures,
        writes_globals,
        writes_persistents,
        may_call_unknown,
        place_mutations,
        calls,
        spawn_safety: SpawnSafetyFact::RequiresIsolation,
        fusibility,
        parallel_safety,
        accel_eligibility,
    };
    store.functions.insert(body.function, summary.clone());
    summary
}

fn classify_fusibility(
    workspace: &[WorkspaceEffect],
    environment: &[runmat_hir::EnvironmentEffect],
    may_call_unknown: bool,
) -> FusibilityFact {
    if may_call_unknown {
        return FusibilityFact::NonFusible("unknown call barrier".into());
    }
    if !environment.is_empty() {
        return FusibilityFact::NonFusible("environment effect barrier".into());
    }
    if workspace
        .iter()
        .any(|effect| !matches!(effect, WorkspaceEffect::None))
    {
        return FusibilityFact::NonFusible("workspace effect barrier".into());
    }
    FusibilityFact::Unknown
}

fn classify_parallel_safety(
    workspace: &[WorkspaceEffect],
    environment: &[runmat_hir::EnvironmentEffect],
    may_call_unknown: bool,
) -> ParallelSafetyFact {
    if may_call_unknown {
        ParallelSafetyFact::Unknown
    } else if !environment.is_empty()
        || workspace
            .iter()
            .any(|effect| !matches!(effect, WorkspaceEffect::None))
    {
        ParallelSafetyFact::WritesSharedState
    } else {
        ParallelSafetyFact::Unknown
    }
}

fn classify_accel_eligibility(
    workspace: &[WorkspaceEffect],
    environment: &[runmat_hir::EnvironmentEffect],
    may_call_unknown: bool,
) -> AccelEligibilityFact {
    if may_call_unknown {
        return AccelEligibilityFact::Ineligible("unknown call barrier".into());
    }
    if !environment.is_empty() {
        return AccelEligibilityFact::Ineligible("environment effect barrier".into());
    }
    if workspace
        .iter()
        .any(|effect| !matches!(effect, WorkspaceEffect::None))
    {
        return AccelEligibilityFact::Ineligible("workspace effect barrier".into());
    }
    AccelEligibilityFact::Unknown
}

fn scan_rvalue(
    body: &MirBody,
    value: &MirRvalue,
    reads_captures: &mut BTreeSet<BindingId>,
    async_behavior: &mut AsyncBehaviorFact,
    calls: &mut Vec<CallSummary>,
    may_call_unknown: &mut bool,
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
            if is_unknown_call(&call.callee) {
                *may_call_unknown = true;
            }
            calls.push(call_summary(call));
            for arg in &call.args {
                scan_operand(body, arg.operand(), reads_captures);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                scan_operand(body, element, reads_captures);
            }
        }
        MirRvalue::Index { base, indexing } => {
            scan_operand(body, base, reads_captures);
            scan_indexing(body, indexing, reads_captures);
        }
        MirRvalue::Future(_) => {}
        MirRvalue::Spawn(future) => {
            *async_behavior = AsyncBehaviorFact::RequiresAsyncRuntime;
            scan_operand(body, future, reads_captures);
        }
    }
}

fn is_unknown_call(callee: &HirCallableRef) -> bool {
    matches!(
        callee,
        HirCallableRef::DynamicExpr(_) | HirCallableRef::Unresolved(_)
    )
}

fn call_summary(call: &MirCall) -> CallSummary {
    CallSummary {
        callee: call.callee.clone(),
        requested_outputs: call.requested_outputs.clone(),
        arg_count: call.args.len(),
        expansion_arg_count: call
            .args
            .iter()
            .filter(|arg| matches!(arg, MirCallArg::Expansion(_)))
            .count(),
    }
}

fn scan_workspace_effect(
    body: &MirBody,
    effect: &WorkspaceEffect,
    bindings: &[MirLocalId],
    writes_globals: &mut BTreeSet<BindingId>,
    writes_persistents: &mut BTreeSet<BindingId>,
) {
    let target = match effect {
        WorkspaceEffect::MutatesGlobal => Some(writes_globals),
        WorkspaceEffect::MutatesPersistent => Some(writes_persistents),
        _ => None,
    };
    if let Some(target) = target {
        for local in bindings {
            if let Some(binding) = local_for_id(body, *local).and_then(|local| local.binding) {
                target.insert(binding);
            }
        }
    }
}

fn scan_place_write(body: &MirBody, place: &MirPlace, writes_captures: &mut BTreeSet<BindingId>) {
    if let Some(local) = place_root(place) {
        scan_local_write(body, local, writes_captures);
    }
}

fn scan_indexing(body: &MirBody, indexing: &MirIndexing, reads_captures: &mut BTreeSet<BindingId>) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) | MirIndexComponent::Logical(operand) => {
                scan_operand(body, operand, reads_captures);
            }
            MirIndexComponent::Colon | MirIndexComponent::End { .. } => {}
        }
    }
}

fn place_root(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root(base)
        }
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
