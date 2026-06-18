use crate::{
    BasicBlockId, MirAssembly, MirBody, MirCallee, MirDiagnostic, MirDiagnosticSeverity,
    MirIndexComponent, MirIndexing, MirLocal, MirLocalId, MirLocalKind, MirOperand, MirPlace,
    MirRvalue, MirStmtKind, MirTerminatorKind, SpawnBoundary,
};
use runmat_hir::{BindingId, FunctionId, Span, SpawnSafetyFact, SpawnSafetyReason};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, PartialEq)]
struct CaptureFacts {
    reads_captures: BTreeSet<BindingId>,
    writes_captures: BTreeSet<BindingId>,
}

fn analyze_capture_facts(body: &MirBody) -> CaptureFacts {
    let mut reads_captures = BTreeSet::new();
    let mut writes_captures = BTreeSet::new();

    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { place, value } => {
                    scan_rvalue(body, value, &mut reads_captures);
                    scan_place_write(body, place, &mut writes_captures);
                }
                MirStmtKind::MultiAssign { targets, value } => {
                    scan_rvalue(body, value, &mut reads_captures);
                    for target in &targets.targets {
                        if let crate::MirOutputTarget::Place(place) = target {
                            scan_place_write(body, place, &mut writes_captures);
                        }
                    }
                }
                MirStmtKind::Expr(value) => {
                    scan_rvalue(body, value, &mut reads_captures);
                }
                MirStmtKind::PlaceMutation(_) => {}
                MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {}
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
                scan_rvalue(body, iterable, &mut reads_captures);
                scan_local_write(body, *binding, &mut writes_captures);
            }
            MirTerminatorKind::Return(return_outputs) => {
                for output in return_outputs {
                    scan_operand(body, output, &mut reads_captures);
                }
            }
            MirTerminatorKind::Await { future, .. } => {
                scan_operand(body, future, &mut reads_captures);
            }
            MirTerminatorKind::Goto(_)
            | MirTerminatorKind::TryCatch { .. }
            | MirTerminatorKind::Unreachable => {}
        }
    }

    CaptureFacts {
        reads_captures,
        writes_captures,
    }
}

fn scan_rvalue(body: &MirBody, value: &MirRvalue, reads_captures: &mut BTreeSet<BindingId>) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) => {
            scan_operand(body, operand, reads_captures);
        }
        MirRvalue::Binary(left, _, right) => {
            scan_operand(body, left, reads_captures);
            scan_operand(body, right, reads_captures);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            scan_operand(body, left, reads_captures);
            for stmt in right_temps {
                match &stmt.kind {
                    crate::MirStmtKind::Assign { value, .. }
                    | crate::MirStmtKind::Expr(value)
                    | crate::MirStmtKind::MultiAssign { value, .. } => {
                        scan_rvalue(body, value, reads_captures);
                    }
                    crate::MirStmtKind::PlaceMutation(_)
                    | crate::MirStmtKind::WorkspaceEffect { .. }
                    | crate::MirStmtKind::EnvironmentEffect(_) => {}
                }
            }
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
            if let MirCallee::Dynamic(callee) = &call.callee {
                scan_operand(body, callee, reads_captures);
            }
            for arg in &call.args {
                scan_operand(body, arg.operand(), reads_captures);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                scan_operand(body, element, reads_captures);
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                scan_operand(body, value, reads_captures);
            }
        }
        MirRvalue::Index { base, indexing } => {
            scan_operand(body, base, reads_captures);
            scan_indexing(body, indexing, reads_captures);
        }
        MirRvalue::Future { args, .. } => {
            for arg in args {
                scan_operand(body, arg.operand(), reads_captures);
            }
        }
        MirRvalue::Member { base, .. } => {
            scan_operand(body, base, reads_captures);
        }
        MirRvalue::DynamicMember { base, member } => {
            scan_operand(body, base, reads_captures);
            scan_operand(body, member, reads_captures);
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. } => {}
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
        MirRvalue::Spawn(future) => {
            scan_operand(body, future, reads_captures);
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
            MirIndexComponent::Expr(operand) => {
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
    match operand {
        MirOperand::Local(local) => {
            if let Some(binding) = capture_binding(body, *local) {
                reads_captures.insert(binding);
            }
        }
        MirOperand::FunctionHandle(_) => {}
        MirOperand::Constant(_) => {}
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

pub(super) fn analyze_assembly_spawn_boundaries(
    assembly: &MirAssembly,
) -> HashMap<FunctionId, Vec<SpawnBoundary>> {
    let capture_facts: HashMap<_, _> = assembly
        .bodies
        .values()
        .map(|body| (body.function, analyze_capture_facts(body)))
        .collect();
    assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            (
                *function,
                analyze_spawn_boundaries_with_capture_facts(body, &capture_facts),
            )
        })
        .collect()
}

fn analyze_spawn_boundaries_with_capture_facts(
    body: &MirBody,
    capture_facts: &HashMap<FunctionId, CaptureFacts>,
) -> Vec<SpawnBoundary> {
    let in_states = future_target_in_states(body);
    let block_index: HashMap<_, _> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut boundaries = Vec::new();
    for block in &body.blocks {
        let mut state = block_index
            .get(&block.id)
            .and_then(|idx| in_states.get(*idx))
            .cloned()
            .unwrap_or_else(|| vec![BTreeSet::new(); body.locals.len()]);
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { place, value } => {
                    collect_classified_spawn_rvalue(
                        value,
                        stmt.span,
                        capture_facts,
                        &state,
                        &mut boundaries,
                    );
                    if let MirPlace::Local(local) = place {
                        state[local.0].clear();
                        if let Some(target) = future_target(value) {
                            state[local.0].insert(target);
                        }
                    }
                }
                MirStmtKind::MultiAssign { value, .. } | MirStmtKind::Expr(value) => {
                    collect_classified_spawn_rvalue(
                        value,
                        stmt.span,
                        capture_facts,
                        &state,
                        &mut boundaries,
                    );
                }
                MirStmtKind::PlaceMutation(_)
                | MirStmtKind::WorkspaceEffect { .. }
                | MirStmtKind::EnvironmentEffect(_) => {}
            }
        }
    }
    boundaries
}

fn future_target_in_states(body: &MirBody) -> Vec<Vec<BTreeSet<FunctionId>>> {
    let local_count = body.locals.len();
    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut in_states = vec![vec![BTreeSet::new(); local_count]; body.blocks.len()];
    let mut out_states = in_states.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for (idx, block) in body.blocks.iter().enumerate() {
            let mut state = in_states[idx].clone();
            transfer_future_targets(block, &mut state);
            if state != out_states[idx] {
                out_states[idx] = state.clone();
                changed = true;
            }
            for successor in successors(&block.terminator.kind) {
                let Some(&successor_idx) = block_index.get(&successor) else {
                    continue;
                };
                for (target, incoming) in in_states[successor_idx].iter_mut().zip(&state) {
                    let before = target.len();
                    target.extend(incoming.iter().copied());
                    if target.len() != before {
                        changed = true;
                    }
                }
            }
        }
    }
    in_states
}

fn transfer_future_targets(block: &crate::BasicBlock, state: &mut [BTreeSet<FunctionId>]) {
    for stmt in &block.statements {
        let MirStmtKind::Assign { place, value } = &stmt.kind else {
            continue;
        };
        if let MirPlace::Local(local) = place {
            state[local.0].clear();
            if let Some(target) = future_target(value) {
                state[local.0].insert(target);
            }
        }
    }
}

pub(super) fn diagnose_spawn_safety(boundaries: &[SpawnBoundary]) -> Vec<MirDiagnostic> {
    boundaries
        .iter()
        .filter_map(|boundary| match &boundary.safety {
            SpawnSafetyFact::NotSpawnSafe { reason } => {
                Some(spawn_safety_diagnostic(reason.clone(), boundary.span))
            }
            SpawnSafetyFact::SpawnSafe | SpawnSafetyFact::RequiresIsolation => None,
        })
        .collect()
}

fn collect_classified_spawn_rvalue(
    value: &MirRvalue,
    span: Span,
    capture_facts: &HashMap<FunctionId, CaptureFacts>,
    future_targets: &[BTreeSet<FunctionId>],
    boundaries: &mut Vec<SpawnBoundary>,
) {
    let MirRvalue::Spawn(future) = value else {
        return;
    };
    let mut boundary = SpawnBoundary {
        future: future.clone(),
        safety: SpawnSafetyFact::RequiresIsolation,
        span,
    };
    boundary.safety = classify_spawn_boundary(&boundary, capture_facts, future_targets);
    boundaries.push(boundary);
}

fn future_target(value: &MirRvalue) -> Option<FunctionId> {
    match value {
        MirRvalue::Future { function, .. } => Some(*function),
        _ => None,
    }
}

fn classify_spawn_boundary(
    boundary: &SpawnBoundary,
    capture_facts: &HashMap<FunctionId, CaptureFacts>,
    future_targets: &[BTreeSet<FunctionId>],
) -> SpawnSafetyFact {
    let Some(targets) = spawn_targets(&boundary.future, future_targets) else {
        return SpawnSafetyFact::NotSpawnSafe {
            reason: SpawnSafetyReason::UnknownDynamicCapture,
        };
    };
    let mut requires_isolation = false;
    for target in targets {
        let Some(summary) = capture_facts.get(&target) else {
            requires_isolation = true;
            continue;
        };
        if !summary.reads_captures.is_empty() || !summary.writes_captures.is_empty() {
            return SpawnSafetyFact::NotSpawnSafe {
                reason: SpawnSafetyReason::MutableLexicalCapture,
            };
        }
    }
    if requires_isolation {
        SpawnSafetyFact::RequiresIsolation
    } else {
        SpawnSafetyFact::SpawnSafe
    }
}

fn spawn_targets(
    future: &MirOperand,
    future_targets: &[BTreeSet<FunctionId>],
) -> Option<BTreeSet<FunctionId>> {
    match future {
        MirOperand::Local(local) => future_targets
            .get(local.0)
            .filter(|targets| !targets.is_empty())
            .cloned(),
        MirOperand::FunctionHandle(_) | MirOperand::Constant(_) => None,
    }
}

fn successors(kind: &MirTerminatorKind) -> Vec<BasicBlockId> {
    match kind {
        MirTerminatorKind::Goto(target) => vec![*target],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, block)| *block)
            .chain(std::iter::once(*otherwise))
            .collect(),
        MirTerminatorKind::For {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            ..
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await { resume, .. } => vec![*resume],
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    }
}

fn spawn_safety_diagnostic(reason: SpawnSafetyReason, span: Span) -> MirDiagnostic {
    let message = match reason {
        SpawnSafetyReason::MutableLexicalCapture => {
            "spawned future mutates a lexical capture from its parent frame"
        }
        SpawnSafetyReason::NonSendableRuntimeHandle => {
            "spawned future captures a non-sendable runtime handle"
        }
        SpawnSafetyReason::UnsynchronizedSharedMutation => {
            "spawned future performs unsynchronized shared mutation"
        }
        SpawnSafetyReason::UnknownDynamicCapture => {
            "spawned future has dynamically captured state with unknown spawn safety"
        }
    };
    MirDiagnostic::new("RM-MIR0003", MirDiagnosticSeverity::Error, message, span)
        .with_primary_label("this spawn crosses an unsafe task boundary")
        .with_help("avoid mutating parent-frame lexical captures from spawned futures")
        .with_category("spawn-safety")
}
