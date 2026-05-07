use crate::{
    AsyncBehaviorFact, MirBody, MirCall, MirCallArg, MirIndexComponent, MirIndexing, MirLocal,
    MirLocalId, MirLocalKind, MirOperand, MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
};
use runmat_hir::{
    AsyncValueFact, BindingId, FunctionHandleTarget, FunctionId, HirCallableRef, PlaceMutation,
    RequestedOutputCount, ShapeFact, SpawnSafetyFact, TypeFact, ValueFlowFact, WorkspaceEffect,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{
    AccelEligibilityFact, AnalysisStore, EffectSummary, FusibilityFact, MirLocalKey,
    ParallelSafetyFact,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionSummary {
    pub function: FunctionId,
    pub abi: FunctionAbiSummary,
    pub outputs: Vec<TypeFact>,
    pub output_shapes: Vec<ShapeFact>,
    pub output_value_flows: Vec<ValueFlowFact>,
    pub output_async_values: Vec<Option<AsyncValueFact>>,
    pub requested_output_sensitive: Vec<(RequestedOutputCount, Vec<TypeFact>)>,
    pub effects: EffectSummary,
    pub reads_captures: BTreeSet<BindingId>,
    pub writes_captures: BTreeSet<BindingId>,
    pub writes_globals: BTreeSet<BindingId>,
    pub writes_persistents: BTreeSet<BindingId>,
    pub may_call_unknown: bool,
    pub place_mutations: Vec<PlaceMutation>,
    pub function_handles: Vec<FunctionHandleTarget>,
    pub calls: Vec<CallSummary>,
    pub spawn_safety: SpawnSafetyFact,
    pub fusibility: FusibilityFact,
    pub parallel_safety: ParallelSafetyFact,
    pub accel_eligibility: AccelEligibilityFact,
}

#[derive(Debug, Clone, PartialEq)]
struct OutputFact {
    ty: TypeFact,
    shape: ShapeFact,
    value_flow: ValueFlowFact,
    async_value: Option<AsyncValueFact>,
}

impl Default for OutputFact {
    fn default() -> Self {
        Self {
            ty: TypeFact::Unknown,
            shape: ShapeFact::Unknown,
            value_flow: ValueFlowFact::UnknownList,
            async_value: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallSummary {
    pub callee: HirCallableRef,
    pub requested_outputs: RequestedOutputCount,
    pub arg_count: usize,
    pub expansion_arg_count: usize,
    pub async_behavior: AsyncBehaviorFact,
    pub dispatch: NominalDispatchHook,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NominalDispatchHook {
    DirectFunction,
    Builtin,
    Constructor,
    MethodSyntax,
    Dynamic,
    OverloadedIndexing,
    None,
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
    let mut outputs = Vec::new();
    let mut function_handles = Vec::new();

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
                        &mut function_handles,
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
                        &mut function_handles,
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
                        &mut function_handles,
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
                scan_operand(body, cond, &mut reads_captures, &mut function_handles);
            }
            MirTerminatorKind::Switch { discr, cases, .. } => {
                scan_operand(body, discr, &mut reads_captures, &mut function_handles);
                for (case, _) in cases {
                    scan_operand(body, case, &mut reads_captures, &mut function_handles);
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
                    &mut function_handles,
                );
                scan_local_write(body, *binding, &mut writes_captures);
            }
            MirTerminatorKind::Return(return_outputs) => {
                output_count = output_count.max(return_outputs.len());
                for (idx, output) in return_outputs.iter().enumerate() {
                    scan_operand(body, output, &mut reads_captures, &mut function_handles);
                    merge_output_fact(&mut outputs, idx, operand_output_fact(body, store, output));
                }
            }
            MirTerminatorKind::Await { future, .. } => {
                async_behavior = AsyncBehaviorFact::RequiresAsyncRuntime;
                scan_operand(body, future, &mut reads_captures, &mut function_handles);
            }
            MirTerminatorKind::Goto(_)
            | MirTerminatorKind::TryCatch { .. }
            | MirTerminatorKind::Unreachable => {}
        }
    }

    let fusibility = classify_fusibility(&workspace, &environment, may_call_unknown);
    let parallel_safety = classify_parallel_safety(&workspace, &environment, may_call_unknown);
    let accel_eligibility = classify_accel_eligibility(&workspace, &environment, may_call_unknown);

    let finalized_outputs = finalize_output_facts(outputs, output_count);

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
        outputs: finalized_outputs
            .iter()
            .map(|fact| fact.ty.clone())
            .collect(),
        output_shapes: finalized_outputs
            .iter()
            .map(|fact| fact.shape.clone())
            .collect(),
        output_value_flows: finalized_outputs
            .iter()
            .map(|fact| fact.value_flow.clone())
            .collect(),
        output_async_values: finalized_outputs
            .iter()
            .map(|fact| fact.async_value.clone())
            .collect(),
        requested_output_sensitive: requested_output_sensitive_facts(&finalized_outputs),
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
        function_handles,
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

fn merge_output_fact(outputs: &mut Vec<Option<OutputFact>>, idx: usize, incoming: OutputFact) {
    if outputs.len() <= idx {
        outputs.resize_with(idx + 1, || None);
    }
    match &mut outputs[idx] {
        Some(existing) => *existing = join_output_fact(existing, &incoming),
        slot @ None => *slot = Some(incoming),
    }
}

fn finalize_output_facts(outputs: Vec<Option<OutputFact>>, output_count: usize) -> Vec<OutputFact> {
    (0..output_count)
        .map(|idx| outputs.get(idx).and_then(Clone::clone).unwrap_or_default())
        .collect()
}

fn operand_output_fact(body: &MirBody, store: &AnalysisStore, operand: &MirOperand) -> OutputFact {
    match operand {
        MirOperand::Local(local) => store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: *local,
            })
            .map(|fact| OutputFact {
                ty: fact.ty.clone(),
                shape: fact.shape.clone(),
                value_flow: fact.value_flow.clone(),
                async_value: fact.async_value.clone(),
            })
            .unwrap_or_default(),
        MirOperand::Constant(crate::MirConstant::Number(_)) => {
            scalar_output_fact(TypeFact::Numeric {
                class: runmat_hir::NumericClass::Double,
                domain: runmat_hir::NumericDomain::Real,
            })
        }
        MirOperand::Constant(crate::MirConstant::String(_)) => {
            scalar_output_fact(TypeFact::CharArray)
        }
        MirOperand::FunctionHandle(runmat_hir::FunctionHandleTarget::Function(function))
        | MirOperand::FunctionHandle(runmat_hir::FunctionHandleTarget::Anonymous(function)) => {
            scalar_output_fact(TypeFact::Function(*function))
        }
        _ => OutputFact::default(),
    }
}

fn scalar_output_fact(ty: TypeFact) -> OutputFact {
    OutputFact {
        ty: ty.clone(),
        shape: ShapeFact::Scalar,
        value_flow: ValueFlowFact::Single(ty),
        async_value: None,
    }
}

fn join_output_fact(left: &OutputFact, right: &OutputFact) -> OutputFact {
    OutputFact {
        ty: join_output_type(&left.ty, &right.ty),
        shape: join_output_shape(&left.shape, &right.shape),
        value_flow: join_output_value_flow(&left.value_flow, &right.value_flow),
        async_value: join_output_async_value(&left.async_value, &right.async_value),
    }
}

fn join_output_type(left: &TypeFact, right: &TypeFact) -> TypeFact {
    if left == right {
        left.clone()
    } else {
        TypeFact::Unknown
    }
}

fn join_output_shape(left: &ShapeFact, right: &ShapeFact) -> ShapeFact {
    if left == right {
        left.clone()
    } else {
        ShapeFact::Unknown
    }
}

fn join_output_value_flow(left: &ValueFlowFact, right: &ValueFlowFact) -> ValueFlowFact {
    if left == right {
        left.clone()
    } else {
        ValueFlowFact::UnknownList
    }
}

fn join_output_async_value(
    left: &Option<AsyncValueFact>,
    right: &Option<AsyncValueFact>,
) -> Option<AsyncValueFact> {
    if left == right {
        left.clone()
    } else {
        None
    }
}

fn scan_rvalue(
    body: &MirBody,
    value: &MirRvalue,
    reads_captures: &mut BTreeSet<BindingId>,
    async_behavior: &mut AsyncBehaviorFact,
    calls: &mut Vec<CallSummary>,
    may_call_unknown: &mut bool,
    function_handles: &mut Vec<FunctionHandleTarget>,
) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) => {
            scan_operand(body, operand, reads_captures, function_handles);
        }
        MirRvalue::Binary(left, _, right) => {
            scan_operand(body, left, reads_captures, function_handles);
            scan_operand(body, right, reads_captures, function_handles);
        }
        MirRvalue::Range { start, step, end } => {
            scan_operand(body, start, reads_captures, function_handles);
            if let Some(step) = step {
                scan_operand(body, step, reads_captures, function_handles);
            }
            scan_operand(body, end, reads_captures, function_handles);
        }
        MirRvalue::Call(call) => {
            if is_unknown_call(&call.callee) {
                *may_call_unknown = true;
            }
            if matches!(async_behavior_for_call(call), AsyncBehaviorFact::MaySuspend)
                && matches!(*async_behavior, AsyncBehaviorFact::NeverSuspends)
            {
                *async_behavior = AsyncBehaviorFact::MaySuspend;
            }
            calls.push(call_summary(call));
            for arg in &call.args {
                scan_operand(body, arg.operand(), reads_captures, function_handles);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                scan_operand(body, element, reads_captures, function_handles);
            }
        }
        MirRvalue::Index { base, indexing } => {
            scan_operand(body, base, reads_captures, function_handles);
            scan_indexing(body, indexing, reads_captures, function_handles);
        }
        MirRvalue::Future { args, .. } => {
            for arg in args {
                scan_operand(body, arg.operand(), reads_captures, function_handles);
            }
        }
        MirRvalue::Member { base, .. } => {
            scan_operand(body, base, reads_captures, function_handles);
        }
        MirRvalue::DynamicMember { base, member } => {
            scan_operand(body, base, reads_captures, function_handles);
            scan_operand(body, member, reads_captures, function_handles);
        }
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
        MirRvalue::Spawn(future) => {
            *async_behavior = AsyncBehaviorFact::RequiresAsyncRuntime;
            scan_operand(body, future, reads_captures, function_handles);
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
        async_behavior: async_behavior_for_call(call),
        dispatch: dispatch_hook(call),
    }
}

fn async_behavior_for_call(call: &MirCall) -> AsyncBehaviorFact {
    if matches!(
        call.callee,
        HirCallableRef::DynamicExpr(_) | HirCallableRef::Unresolved(_)
    ) {
        AsyncBehaviorFact::MaySuspend
    } else {
        AsyncBehaviorFact::NeverSuspends
    }
}

fn dispatch_hook(call: &MirCall) -> NominalDispatchHook {
    match (&call.callee, &call.syntax) {
        (HirCallableRef::Function(_), _) => NominalDispatchHook::DirectFunction,
        (HirCallableRef::Builtin(_), _) => NominalDispatchHook::Builtin,
        (HirCallableRef::ClassConstructor(_), _) => NominalDispatchHook::Constructor,
        (_, runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke) => {
            NominalDispatchHook::MethodSyntax
        }
        (HirCallableRef::DynamicExpr(_) | HirCallableRef::Unresolved(_), _) => {
            NominalDispatchHook::Dynamic
        }
        _ => NominalDispatchHook::None,
    }
}

fn requested_output_sensitive_facts(
    outputs: &[OutputFact],
) -> Vec<(RequestedOutputCount, Vec<TypeFact>)> {
    (0..=outputs.len())
        .map(|count| {
            (
                RequestedOutputCount::Exactly(count),
                outputs
                    .iter()
                    .take(count)
                    .map(|fact| fact.ty.clone())
                    .collect(),
            )
        })
        .collect()
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

fn scan_indexing(
    body: &MirBody,
    indexing: &MirIndexing,
    reads_captures: &mut BTreeSet<BindingId>,
    function_handles: &mut Vec<FunctionHandleTarget>,
) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) | MirIndexComponent::Logical(operand) => {
                scan_operand(body, operand, reads_captures, function_handles);
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

fn scan_operand(
    body: &MirBody,
    operand: &MirOperand,
    reads_captures: &mut BTreeSet<BindingId>,
    function_handles: &mut Vec<FunctionHandleTarget>,
) {
    match operand {
        MirOperand::Local(local) => {
            if let Some(binding) = capture_binding(body, *local) {
                reads_captures.insert(binding);
            }
        }
        MirOperand::FunctionHandle(target) => {
            if !function_handles.contains(target) {
                function_handles.push(target.clone());
            }
        }
        MirOperand::Temp(_) | MirOperand::Constant(_) => {}
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
